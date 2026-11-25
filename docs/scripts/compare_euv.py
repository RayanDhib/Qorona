"""Compare a Qorona EUV frame with an observed EUV image.

Usage:  python compare_euv.py MODEL.fits OBSERVED.fits

MODEL.fits comes from `qorona euv --export fits`. OBSERVED.fits is an image of the same channel:
AIA Level 1, EUVI Level 1 from secchi_prep, or EUI/FSI Level 2. The script puts the observed image
in the unit of the model for each second, corrects AIA for degradation if aiapy is installed, and
averages it onto the pixels of the model frame. It writes MODEL_vs_observed.png: the two images,
each on its own display with its colour bar, and the median of each ring of distance from Sun
centre. It also prints the median ratio model / observed on the disk and in bands off the limb.
"""

import sys
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import sunpy.map
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sunpy.coordinates.screens import SphericalScreen

model_path, observed_path = Path(sys.argv[1]), Path(sys.argv[2])
model = sunpy.map.Map(model_path)
observed = sunpy.map.Map(observed_path)
# The channel of the model, for example AIA-193, EUVI-A-195 or EUI-FSI-174: the observed image must
# come from the same instrument, the same spacecraft and the same wavelength.
channel = model.meta["detector"]
same = (
    observed.detector.upper() == {"AIA": "AIA", "EUVI": "EUVI", "EUI": "FSI"}[channel.split("-")[0]]
)
same &= round(observed.wavelength.to_value(u.AA)) == int(model.meta["wavelnth"])
if channel.startswith("EUVI"):
    same &= observed.observatory.upper().endswith(channel.split("-")[1])
if not same:
    sys.exit(f"The observed image is not in the channel of the model ({channel}).")
hours = abs((observed.date - model.date).to_value(u.h))
if hours > 1.0:
    print(f"warning: the images are {hours:.1f} h apart; the Sun turns 0.55 degree each hour")

# The observed image in the unit of the model for each second.
notes = []
if observed.instrument.upper().startswith("AIA"):
    try:
        from aiapy.calibrate import correct_degradation
    except ImportError:
        notes.append("AIA not corrected for degradation: install aiapy.")
        print(f"warning: {notes[-1]}")
    else:
        observed = correct_degradation(observed)
    per_second = observed.data / observed.exposure_time.to_value(u.s)
else:
    observed_unit = str(observed.meta.get("bunit", "")).lower().replace(" ", "")
    if not observed_unit.endswith(("/s", "s-1", "/sec")):
        sys.exit(
            f"The observed image is in '{observed_unit}', not for each second. "
            "Use EUVI Level 1 from secchi_prep (photons per second) or EUI/FSI Level 2 (DN/s)."
        )
    per_second = observed.data
observed = sunpy.map.Map(per_second.astype(float), observed.meta)
observed.meta["rsun_ref"] = model.rsun_meters.to_value(u.m)  # one solar radius for both images

# Rings of 0.02 solar radius on the observed pixels. The comparison stops at the first ring above
# the limb where more than 10 % of the observed pixels are zero or negative: beyond it, the image is
# outside the field of the instrument or holds only noise.
step = 0.02
centre = observed.wcs.world_to_pixel(
    SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=observed.coordinate_frame)
)
rows, columns = np.indices(observed.data.shape)
angle = np.hypot(columns - centre[0], rows - centre[1]) * observed.scale[0].to_value(u.rad / u.pix)
ring_of_pixel = ((observed.dsun * np.sin(angle)).to_value(u.R_sun) / step).astype(int).ravel()
positive = np.bincount(ring_of_pixel, weights=(observed.data > 0).ravel())
fraction = positive / np.maximum(np.bincount(ring_of_pixel), 1)
noisy = np.nonzero((fraction < 0.9) & (np.arange(fraction.size) * step >= 1.0))[0]
edge = noisy[0] * step if noisy.size else np.inf

# Average the observed pixels onto the model pixels. Points off the disk are on a sphere around
# the observer, so that the corona above the limb is kept.
with SphericalScreen(observed.observer_coordinate, only_off_disk=True):
    resampled = observed.reproject_to(model.wcs, algorithm="adaptive", conserve_flux=False)
model_data = model.data.astype(float)
observed_data = resampled.data

# Distance from Sun centre of the closest point of each line of sight, in solar radii.
coordinates = sunpy.map.all_coordinates_from_map(model)
angle = np.arccos(np.cos(coordinates.Tx) * np.cos(coordinates.Ty))
impact = (model.dsun * np.sin(angle)).to_value(u.R_sun)
valid = np.isfinite(observed_data) & (observed_data > 0) & np.isfinite(model_data)
valid &= (model_data > 0) & (impact < edge)

with np.errstate(divide="ignore", invalid="ignore"):
    log_ratio = np.where(valid, np.log10(model_data / observed_data), np.nan)
bands = {"disk": (0.0, 1.0), "1.0-1.1": (1.0, 1.1), "1.1-1.3": (1.1, 1.3), "1.3-2.0": (1.3, 2.0)}
lines = []
for name, (low, high) in bands.items():
    band = valid & (impact >= low) & (impact < high)
    if band.sum() >= 100:
        lines.append(f"{name} R_sun: model / observed = {10 ** np.median(log_ratio[band]):.3g}")
if np.isfinite(edge):
    lines.append(f"compared out to {edge:.2f} R_sun")
print("\n".join(lines))

# Each image on its own display, 0.5 to 99.9 percent, with its colour bar, so that both show their
# structure. The profile compares the levels.
unit = model.meta["bunit"]
figure = plt.figure(figsize=(13, 10))
grid = figure.add_gridspec(2, 2, height_ratios=(1, 0.6), hspace=0.25, wspace=0.15)
date = observed.date.isot[:19]
for column, (title, data) in enumerate(
    ((f"Observed {channel}, {date}", observed_data), (f"Model {channel}", model_data))
):
    low, high = np.percentile(data[valid], [0.5, 99.9])
    axes = figure.add_subplot(grid[0, column])
    shown = axes.imshow(
        np.where(valid, data, np.nan),
        norm=LogNorm(vmin=low, vmax=high, clip=True),
        cmap=observed.cmap,
        origin="lower",
    )
    axes.set_title(title, fontsize=10)
    axes.set_axis_off()
    figure.colorbar(shown, ax=axes, fraction=0.046, pad=0.02, label=unit)

# The median of each ring.
edges = np.arange(0.0, impact[valid].max() + step, step)
index = np.digitize(impact, edges)
middles = 0.5 * (edges[:-1] + edges[1:])
axes = figure.add_subplot(grid[1, :])
for title, data, color in (("observed", observed_data, "black"), ("model", model_data, "tab:red")):
    medians = np.full(middles.size, np.nan)
    for i in range(1, edges.size):
        ring = valid & (index == i)
        if ring.sum() >= 30:
            medians[i - 1] = np.median(data[ring])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # empty rings are not drawn
        axes.semilogy(middles, medians, color=color, label=title)
axes.axvline(1.0, color="gray", linewidth=0.8)
axes.set_xlabel("distance from Sun centre (solar radii)")
axes.set_ylabel(f"median of each ring ({unit})")
axes.grid(True, which="both", alpha=0.3)
axes.legend(fontsize=9)
notes.append(
    "The model has no scattered light (PSF): dark regions of the observation are brighter."
)
figure.text(0.5, 0.01, "  ".join(lines) + "\n" + " ".join(notes), ha="center", fontsize=9)
output = model_path.with_name(f"{model_path.stem}_vs_observed.png")
figure.savefig(output, dpi=100, bbox_inches="tight")
print(output)
