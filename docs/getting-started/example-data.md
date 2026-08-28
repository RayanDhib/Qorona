# Example data

Pick a working directory; every command in these docs runs from it, with the data in `data/`
beneath it.

The quickstart and every figure in this documentation use `coconut_corona.CFmesh.xz` (~165 MB), a
COCONUT coronal MHD solution. It is distributed as a release asset, not committed to
the repository: download it into `data/` before running the commands.

```bash
mkdir -p data
curl -L -o data/coconut_corona.CFmesh.xz \
    https://github.com/RayanDhib/Qorona/releases/download/v0.4.0/coconut_corona.CFmesh.xz
```

Qorona reads the compressed `.xz` directly; there is no manual decompression step. Building a
volume from it writes a ~0.4 GB `.qor` beside the input (more at `--quality high`; see
[the Q⊥ volume](../qperp-volume.md)).

The solution was run in the Carrington frame, and its observation time is `2025-10-09T18:19:52`.
The two are passed to the commands as `--frame carrington` and `--timestamp`: the time stamps
images with the correct Carrington rotation, and together they let the camera take a real
observer's viewpoint. Both flags are optional; without them the stamp omits the Carrington
rotation and Julian date, and the viewpoint needs to be set manually.
