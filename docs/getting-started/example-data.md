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

The solution's observation time is `2025-10-09T18:19:52`, passed to the commands as
`--timestamp` so images are stamped with the correct Carrington rotation. The flag is optional:
without it the image stamp simply omits the Carrington rotation and Julian date.
