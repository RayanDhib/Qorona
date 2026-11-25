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

Expect one warning in this example: the NFI frame reaches past the 25 solar radii shell of the
solution (see [The two frames](punch-details.md#the-two-frames)).

| Flag | Use |
|------|-----|
| `--instrument` | The frame of the PUNCH file: `punch-mosaic` for PAM (polarized mosaic) and CAM (clear mosaic), `punch-nfi` for CNN (clear NFI image). |
| `--punch-l3` | Subtract the background as PUNCH Level 3 does. The observer is Earth. |
| `--export fits` | Write the FITS file that the comparison script reads. |
| `--timestamp`, `--frame` | The time and the frame of a coronal solution. See [Solution time and frame](time-and-frame.md). |
| `--magnetogram-time`, `--relaxation` | The run settings of an Icarus snapshot, or `--par`. See [Icarus run settings](heliospheric.md#icarus-run-settings). |

The model frames must be in mean solar brightness (MSB), the unit of PUNCH. See
[Brightness units](products/white-light.md#brightness-units).

[PUNCH comparison details](punch-details.md) describes the two frames, the observer, the Level 3
treatment and the FITS output.

## The comparison script

Download [`compare_punch.py`](scripts/compare_punch.py). It needs only NumPy, Astropy and
Matplotlib.

??? example "compare_punch.py"
    ```python
    --8<-- "scripts/compare_punch.py"
    ```

The figure shows PUNCH and the model with the Level 3 treatment, with the same display. Below them,
a profile gives the median brightness in each ring of elongation. The model without the treatment
(the full K corona) is in the `PB_FULL` and `TOTAL_FULL` extensions of the FITS file.

The profile uses rings of elongation: 1 degree for the mosaics, 0.25 degree for NFI. Elongation is
the angle at the observer between the line to Sun centre and the line to the point. From Earth, a
line of sight at elongation ε passes the Sun at 215 × sin ε solar radii.
