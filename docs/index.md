# Qorona {.qhide}

<div class="qhero">
  <img class="qhero-bg" src="assets/hero-eclipse.jpg" alt="">
  <div class="qhero-fade"></div>
  <div class="qhero-inner">
    <p class="qhero-wm">Qorona<span class="qhero-perp">&#x22A5;</span></p>
    <p class="qhero-tag">Synthetic coronal imagery from global MHD solutions</p>
    <div class="qhero-rule"></div>
  </div>
</div>

Qorona turns a global coronal **MHD solution** into **eclipse-like synthetic imagery**. Its
primary product is a line-of-sight integral of the perpendicular magnetic **squashing factor
Q⊥**, the quantity that lights up the thin loops, streamers, and current sheets seen at a
total solar eclipse. The result is rendered for morphological comparison against eclipse and
coronagraph observations.

```
coronal MHD solution ──▶ read ──▶ resample ──▶ Q⊥ volume ──▶ LOS render ──▶ synthetic eclipse image
```

![Synthetic eclipse render of the COCONUT corona](assets/eclipse.png)

??? note "How this image was made"
    The [example solution](getting-started/example-data.md), built once and rendered once:

    ```bash
    qorona build data/coconut_corona.CFmesh.xz -o data/coconut_corona.qor \
        --timestamp 2025-10-09T18:19:52 --outer-radius 8
    qorona render data/coconut_corona.qor -o data/eclipse.png \
        --fov 8 --longitude 317 --latitude 6.2
    ```

**Get started** with the [installation](getting-started/installation.md), grab the
[example data](getting-started/example-data.md), and make your
[first eclipse image](getting-started/first-eclipse-image.md).

**Products**: the [squashing-factor render](products/squashing-factor.md), the
[polarity view](products/polarity.md), [white-light imaging](products/white-light.md),
[Q-maps](products/qmaps.md), and [field lines](products/fieldlines.md), each with the exact
command that produced its figure.

**How it runs**: [the Q⊥ volume](qperp-volume.md) (build once, render many),
[GPU acceleration](gpu.md), and, on clusters, the [HPC guide](getting-started/hpc.md).
