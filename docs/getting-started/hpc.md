# HPC clusters

Qorona installs from PyPI; pick whichever your cluster makes easiest.

**With pip**, into a plain virtual environment:

```bash
module purge            # optional; stops other modules leaking packages via PYTHONPATH
module load python      # any Python >= 3.11, e.g. Python/3.12.3-GCCcore-13.3.0
python -m venv ~/envs/qorona && source ~/envs/qorona/bin/activate
pip install qorona
```

**With conda**, if you prefer it or a binary dependency (e.g. pyhdf/HDF4) is easier from
conda-forge:

```bash
conda create -n qorona -c conda-forge python=3.11
conda activate qorona
pip install qorona
```

Two flags are made for batch jobs: `--workers` pins the kernel threads to your allocation
(numba otherwise takes every core it sees), and `--quiet` keeps job logs readable.

!!! note "One node, one task, many CPUs"
    Qorona is shared-memory threaded (numba), not MPI: request a full node's cores with
    `--cpus-per-task` and do not use `mpirun`.

On GPU nodes, load the site's CUDA module (e.g. `module load CUDA/12.4.0`) so numba finds the
toolkit; `--device gpu` errors loudly if the GPU is unusable rather than silently falling back. See
[GPU acceleration](../gpu.md) for the backend details and timings.

## SLURM templates

Both templates live in
[`hpc/`](https://github.com/RayanDhib/Qorona/tree/main/hpc) in the repository;
replace the `<placeholders>` with your site's values. The `--cluster` line is only needed where
one scheduler serves several clusters (e.g. VSC); drop it otherwise.

### CPU node

```bash
#!/bin/bash -l
#SBATCH --job-name=qorona-cpu
#SBATCH --cluster=<cluster>
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH -o qorona-cpu-%j.out
#SBATCH -e qorona-cpu-%j.err

# Qorona is threaded (numba), not MPI: one node, one task, many cores.
module purge
module load Python/3.12.3-GCCcore-13.3.0    # adjust to your site: `module spider Python`
source <path-to-venv>/bin/activate          # the venv Qorona is in (or: conda activate qorona)

cd <case-directory>

# Build the volume once, then render any number of viewpoints off it.
# --workers ties the threads to the cores SLURM gave this job.
qorona build solution.CFmesh -o solution.qor --outer-radius 8 \
    --device cpu --workers "$SLURM_CPUS_PER_TASK" --quiet

qorona render solution.qor -o eclipse.png --fov 8 --longitude 317 --latitude 6.2 \
    --workers "$SLURM_CPUS_PER_TASK" --quiet
```

### GPU node

```bash
#!/bin/bash -l
#SBATCH --job-name=qorona-gpu
#SBATCH --cluster=<cluster>
#SBATCH --partition=<gpu-partition>
#SBATCH --account=<account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:15:00
#SBATCH -o qorona-gpu-%j.out
#SBATCH -e qorona-gpu-%j.err

# One GPU builds the volume; a few CPU cores feed it. One node, one task.
module purge
module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.4.0    # adjust versions to your site
source <path-to-venv>/bin/activate                     # the venv Qorona is in (or: conda activate qorona)

cd <case-directory>

# --device gpu builds on the GPU (it errors loudly if the GPU is unusable).
qorona build solution.CFmesh -o solution.qor --outer-radius 8 \
    --device gpu --workers "$SLURM_CPUS_PER_TASK" --quiet

qorona render solution.qor -o eclipse.png --fov 8 --longitude 317 --latitude 6.2 \
    --workers "$SLURM_CPUS_PER_TASK" --quiet
```
