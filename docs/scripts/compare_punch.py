"""Compare a Qorona frame with a PUNCH Level 3 image.

Usage:  python compare_punch.py MODEL.fits PUNCH.fits [pb|total]

MODEL.fits comes from `qorona wl --punch-l3 --export fits` with the instrument of the PUNCH file.
The script writes MODEL_vs_punch.png: PUNCH and the treated model with the same display, and below
them the median brightness of each ring of elongation.
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


images = {"PUNCH": blocks(punch), "model": blocks(treated)}

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
figure = plt.figure(figsize=(12, 9))
grid = figure.add_gridspec(2, 2, height_ratios=(1, 0.55), hspace=0.25, wspace=0.06)
titles = {
    "PUNCH": f"PUNCH L3 {product} {name}, {date}",
    "model": f"Model {name}, PUNCH L3 treatment",
}
for column, key in enumerate(images):
    axes = figure.add_subplot(grid[0, column])
    axes.imshow(np.nan_to_num(images[key]).clip(0), cmap="gray", norm=norm, origin="lower")
    axes.set_title(titles[key], fontsize=10)
    axes.set_axis_off()
axes = figure.add_subplot(grid[1, :])
middles = 0.5 * (edges[:-1] + edges[1:])
for key, color in {"PUNCH": "black", "model": "tab:red"}.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # empty or negative rings are not drawn
        axes.semilogy(middles, ring_medians(images[key]), color=color, label=titles[key])
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
