# Comparing with PUNCH

PUNCH (Polarimeter to Unify the Corona and Heliosphere) is a NASA mission. It images the outer
corona and the young solar wind in polarized white light (DeForest et al. 2026). Qorona renders a
model as a PUNCH Level 3 image: the same frame, the same unit and the same background subtraction.
You can then compare the two pixel for pixel.

## Quick start

1. Download a PUNCH Level 3 file from the
   [PUNCH data archive](https://umbra.nascom.nasa.gov/punch/3/). Pick the file whose exposure
   (`DATE-BEG` to `DATE-END`) holds the solution time.
2. Render the model in the frame of that file, with the
   [PUNCH Level 3 treatment](punch-details.md#the-punch-level-3-treatment).
3. Compare the two with the [comparison script](#the-comparison-script).

An Icarus snapshot, compared with a PUNCH polarized mosaic (PAM):

```bash
qorona wl data/icarus_snapshot.vtu -o icarus_l3.png --instrument punch-mosaic --punch-l3 \
    --magnetogram-time 2025-09-10T00:04:00 --relaxation 0 --export fits
python compare_punch.py icarus_l3.fits PUNCH_L3_PAM_20250923025600_v0m.fits
```

A coronal solution, compared with a clear image of the PUNCH Narrow Field Imager (NFI, product CNN):

```bash
qorona wl data/coconut_corona.CFmesh.xz -o coconut_nfi.png --instrument punch-nfi --punch-l3 \
    --outer-radius 25 --timestamp 2025-10-09T18:19:52 --frame carrington --export fits
python compare_punch.py coconut_nfi.fits PUNCH_L3_CNN_20251009181621_v0m.fits
```

| Flag | Use |
|------|-----|
| `--instrument` | The frame of the PUNCH file: `punch-mosaic` for PAM and CAM, `punch-nfi` for CNN. |
| `--punch-l3` | Subtract the background as PUNCH Level 3 does. The observer is Earth. |
| `--export fits` | Write the FITS file that the comparison script reads. |
| `--timestamp`, `--frame` | The solution time and frame of a coronal solution. See [The observer](punch-details.md#the-observer). |
| `--magnetogram-time`, `--relaxation` | The run settings of an Icarus snapshot, or `--par`. See [Icarus run settings](heliospheric.md#icarus-run-settings). |

The model frames must be in mean solar brightness (MSB), the unit of PUNCH. See
[Brightness units](products/white-light.md#brightness-units).

[PUNCH comparison details](punch-details.md) describes the two frames, the observer, the Level 3
treatment and the FITS output.

## The comparison script

Save this script as `compare_punch.py`. It needs only NumPy, Astropy and Matplotlib.

```python
"""Compare a Qorona frame with a PUNCH Level 3 image.

Usage:  python compare_punch.py MODEL.fits PUNCH.fits [pb|total]

MODEL.fits comes from `qorona wl --punch-l3 --export fits` with the instrument of the PUNCH file.
The script writes MODEL_vs_punch.png: PUNCH, the treated model and the untreated model with the
same display, and below them the median brightness of each ring of elongation.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm

model_path, punch_path = Path(sys.argv[1]), Path(sys.argv[2])
with fits.open(punch_path) as hdus:
    product = hdus[1].header["TYPECODE"] + hdus[1].header["OBSCODE"]
    date = hdus[1].header["DATE-AVG"][:16]
    data = hdus[1].data.astype(float)
component = sys.argv[3] if len(sys.argv) > 3 else ("pb" if product == "PAM" else "total")
if product == "PAM":
    punch = data[1] if component == "pb" else data[0]  # layers: tB, pB, pB'
elif component == "total":
    punch = data
else:
    sys.exit(f"{product} holds total brightness only; use a PAM file for pB")
layer = "PB" if component == "pb" else "TOTAL"
with fits.open(model_path) as hdus:
    treated = hdus[layer].data.astype(float)
    full = hdus[f"{layer}_FULL"].data.astype(float)
    wcs = WCS(hdus[layer].header)
    if hdus[layer].header.get("BUNIT") != "2.009e+07 W/(m2 sr)":
        sys.exit("The model frames are relative, not in MSB; they cannot be compared with PUNCH.")
if punch.shape != treated.shape:
    sys.exit("The images differ in size; render with the instrument of the PUNCH file.")

# Average blocks of n x n valid pixels, so that each image is 512 pixels across.
n = punch.shape[0] // 512
valid = np.isfinite(punch) & (punch != 0) & (treated != 0)


def blocks(image):
    good = valid.reshape(512, n, 512, n)
    total = np.where(good, image.reshape(512, n, 512, n), 0).sum(axis=(1, 3))
    count = good.sum(axis=(1, 3))
    with np.errstate(invalid="ignore"):
        return np.where(count > n * n // 2, total / count, np.nan)


images = {"PUNCH": blocks(punch), "treated": blocks(treated), "untreated": blocks(full)}

# Elongation: the angle at the observer between Sun centre and the pixel.
centres = np.arange(512) * n + (n - 1) / 2
lon, lat = wcs.pixel_to_world_values(*np.meshgrid(centres, centres))
elongation = np.degrees(np.arccos(np.cos(np.radians(lon)) * np.cos(np.radians(lat))))
ring = 0.25 if product.endswith("N") else 1.0  # NFI: 0.25 degree rings; mosaics: 1 degree
seen = np.isfinite(images["PUNCH"])
edges = np.arange(np.floor(elongation[seen].min()), elongation[seen].max() + ring, ring)
index = np.digitize(elongation, edges)


def ring_medians(image):
    out = np.full(edges.size - 1, np.nan)
    for i in range(1, edges.size):
        values = image[(index == i) & np.isfinite(image)]
        if values.size >= 30:
            out[i - 1] = np.median(values)
    return out


# The PUNCH quicklook display: gamma 1/2.2 from 0 to the 99th percentile of the PUNCH pixels.
norm = PowerNorm(gamma=1 / 2.2, vmin=0, vmax=np.percentile(images["PUNCH"][seen], 99))
name = "pB" if component == "pb" else "total brightness"
figure = plt.figure(figsize=(17, 10.5))
grid = figure.add_gridspec(2, 3, height_ratios=(1, 0.55), hspace=0.25, wspace=0.06)
titles = {
    "PUNCH": f"PUNCH L3 {product} {name}, {date}",
    "treated": f"Model {name}, PUNCH L3 treatment",
    "untreated": f"Model {name}, full K corona",
}
for column, key in enumerate(images):
    axes = figure.add_subplot(grid[0, column])
    axes.imshow(np.nan_to_num(images[key]).clip(0), cmap="gray", norm=norm, origin="lower")
    axes.set_title(titles[key], fontsize=10)
    axes.set_axis_off()
axes = figure.add_subplot(grid[1, :])
middles = 0.5 * (edges[:-1] + edges[1:])
styles = {"PUNCH": ("black", "-"), "treated": ("tab:red", "-"), "untreated": ("tab:red", "--")}
for key, (color, line) in styles.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # empty or negative rings are not drawn
        axes.semilogy(middles, ring_medians(images[key]), color=color, ls=line, label=titles[key])
axes.set_xlabel("elongation (degrees)")
axes.set_ylabel(f"{name}, median of each ring (MSB)")
axes.grid(True, which="both", alpha=0.3)
axes.legend(fontsize=9)
if component == "total":
    figure.text(
        0.5,
        0.01,
        "PUNCH Level 3 total brightness carries extra unpolarized light, so read this comparison "
        "with care; prefer pB when you can.",
        ha="center",
        style="italic",
    )
output = model_path.with_name(f"{model_path.stem}_vs_punch.png")
figure.savefig(output, dpi=100, bbox_inches="tight")
print(output)
```

The figure shows:

- **Three images with the same display:** PUNCH, the treated model and the untreated model (the
  full K corona).
- **A profile:** the median brightness in each ring of elongation (1 degree for the mosaics, 0.25
  degree for NFI).

Elongation is the angle at the observer between the line to Sun centre and the line to the point.
From Earth, a line of sight at elongation ε passes the Sun at 215 × sin ε solar radii.

- Compare the treated model with PUNCH. The untreated model shows what the treatment removes.
- PUNCH Level 3 total brightness carries extra unpolarized light, so read a total-brightness
  comparison with care. Prefer polarized brightness (pB) when you can.
