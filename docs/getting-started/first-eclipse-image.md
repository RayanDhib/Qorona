# Quick start

Make a synthetic eclipse image from an example solution in four steps.
[Install Qorona](installation.md) first.

## 1. Download the example data

Work in one directory. The commands in these docs read the data from `data/` in that directory.

```bash
mkdir -p data
curl -L -o data/coconut_corona.CFmesh.xz \
    https://github.com/RayanDhib/Qorona/releases/download/v0.4.0/coconut_corona.CFmesh.xz
```

The file is a COCONUT coronal solution of about 165 MB. Qorona reads the `.xz` file directly. All
the figures in these docs come from this solution. Its time is `2025-10-09T18:19:52` and its frame
is Carrington: the commands below give them with `--timestamp` and `--frame`.

## 2. Inspect the solution

```bash
qorona info data/coconut_corona.CFmesh.xz --timestamp 2025-10-09T18:19:52
```

`info` shows the model, the mesh, the variables and the boundaries. It does not render. Use it to
make sure that Qorona reads your file.

## 3. Build the Q⊥ volume

```bash
qorona build data/coconut_corona.CFmesh.xz -o data/coconut_corona.qor \
    --timestamp 2025-10-09T18:19:52 --frame carrington --outer-radius 8
```

The build takes several minutes and writes `data/coconut_corona.qor` (about 0.4 GB). Build once
for each solution. For a quicker first look, add `--quality fast`.

- `--timestamp` and `--frame` give the time and the frame of the solution. See
  [Solution time and frame](../time-and-frame.md).
- `--outer-radius` sets how far out the volume goes. See [The Q⊥ volume](../qperp-volume.md).

## 4. Render

```bash
qorona render data/coconut_corona.qor -o data/eclipse.png --fov 8 --observer earth
```

![Synthetic eclipse render of the COCONUT corona](../assets/eclipse.png)

`--observer earth` puts the camera at Earth at the solution time. A render takes seconds. Render
the same volume from as many viewpoints as you want, for example from STEREO-A:

```bash
qorona render data/coconut_corona.qor -o data/stereo-a.png --fov 8 --observer stereo-a
```

[Camera and viewpoint](../camera.md) gives all the camera flags.

## What you get

- Each command ends with a summary of its parameters and results.
- The PNG has a stamp in one corner: Carrington rotation, time, viewpoint, roll and field of view.

`qorona run` builds and renders in one command. `--save-volume` keeps the volume for more renders:

```bash
qorona run data/coconut_corona.CFmesh.xz -o data/eclipse.png \
    --timestamp 2025-10-09T18:19:52 --frame carrington --outer-radius 8 \
    --fov 8 --observer earth --save-volume data/coconut_corona.qor
```

## Next

Each product has its own page: [squashing-factor render](../products/squashing-factor.md),
[polarity view](../products/polarity.md), [white-light imaging](../products/white-light.md),
[Q-maps](../products/qmaps.md) and [field lines](../products/fieldlines.md).

To use your own solution, see [Supported models](supported-models.md) and
[Solution time and frame](../time-and-frame.md).
