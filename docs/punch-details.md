# PUNCH comparison details

This page describes the details of the PUNCH comparison: the instrument frames, the observer, the
Level 3 background subtraction and the FITS output. PUNCH is the Polarimeter to Unify the Corona
and Heliosphere. pB is the polarized brightness. For the commands, see
[Comparing with PUNCH](punch.md).

## The two frames

PUNCH has three Wide Field Imagers (WFI) and one Narrow Field Imager (NFI).

| `--instrument` | PUNCH products | Projection | Plate scale | Pixels | Width |
|----------------|----------------|------------|-------------|--------|-------|
| `punch-mosaic` | PAM (polarized mosaic: total brightness, pB, pB'), CAM (clear mosaic: total brightness) | ARC | 0.0225 degrees | 4096 x 4096 | 92.16 degrees |
| `punch-nfi` | CNN (clear NFI image: total brightness) | TAN | 30 arcsec | 2048 x 2048 | 17.07 degrees |

- An instrument frame always uses the [perspective camera](camera.md#two-cameras).
- Qorona uses the reference pixel of PUNCH, so the pixels align one to one.
- `--fov-deg`, `--width` and `--height` override the frame. The frame then no longer matches PUNCH.
- The solution shell must reach as far as the frame. The default coronal shell ends at 12.5 solar
  radii, 3.3 degrees from Earth. Qorona warns when the frame reaches past the shell. The NFI
  example of the [quick start](punch.md#quick-start) raises `--outer-radius` to 25 solar radii
  (6.7 degrees), the outer boundary of that COCONUT solution. The NFI frame reaches 8.5 degrees,
  so the example still warns.
- The NFI frame covers about 1.5 to 8.5 degrees from Sun centre (5.6 to 32 solar radii seen from
  1 astronomical unit (AU)), so it suits coronal solutions. A heliospheric solution with its inner
  boundary at 0.1 AU (5.7 degrees) barely overlaps it.

## The observer

PUNCH observes from Earth orbit. `--punch-l3` places the camera at Earth at the solution time.
Without `--punch-l3`, use `--observer earth`.

- **Coronal solutions** (COCONUT, MAS): give `--timestamp` and `--frame`. See
  [Fetching the viewpoint from an observer](camera.md#fetching-the-viewpoint-from-an-observer).
- **Icarus solutions**: give the run settings of the snapshot instead of `--timestamp` and
  `--frame`. See [Icarus run settings](heliospheric.md#icarus-run-settings).

## The PUNCH Level 3 treatment

PUNCH removes a background from each Level 3 image. For each pixel, it subtracts a value near the
faintest over about 32 days (the 2nd percentile). A plain `qorona wl` frame keeps the full corona
of the model, so it is not comparable with Level 3.

`--punch-l3` applies the same subtraction to the model:

1. Qorona renders the model from Earth every 12 hours, from 16 days before to 16 days after the
   solution time.
2. For each pixel, it takes the 2nd percentile of these 65 frames as the background.
3. It subtracts the background from the frame at the solution time.

Qorona does this separately for each of the three polarizer channels, before it forms pB.

- The solution is one snapshot, so the model background changes only with the rotation of the Sun.
- The PNG uses the PUNCH quicklook display: a gamma 1/2.2 stretch from 0 to the 99th percentile.
  `--punch-l3` refuses `--vignette`, `--mgn`, `--scaling` and `--percentiles`.
- The run takes about 4 minutes for the mosaic on a laptop.
- Near the dark inner edge of the model, the subtraction is not reliable. Qorona sets a band of up
  to 0.36 degrees (mosaic) to 0 there.
- The frames are the K corona of the solution: Thomson scattering by its electrons. They contain no
  F corona, stray light, point-spread function or instrument vignetting.

## FITS output

`--export fits` writes the world coordinate cards of PUNCH: the projection, the plate scale, the
reference pixel and the observer position. If the frames are in mean solar brightness (MSB), the
raw extensions carry the PUNCH unit, `BUNIT = '2.009e+07 W/(m2 sr)'`.

With `--punch-l3`, the extensions are:

| Extension | Content |
|-----------|---------|
| `PB`, `TOTAL` | pB and total brightness after the treatment |
| `PB_FULL`, `TOTAL_FULL` | pB and total brightness without the treatment (the full K corona) |

The primary header records the treatment in the `L3WINDOW`, `L3STEP`, `L3PCTL`, `L3FRAMES` and
`L3BKGPIX` cards.
