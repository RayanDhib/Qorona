# HPC clusters

## Install

Install Qorona from PyPI into a plain virtual environment:

```bash
module purge            # optional: keeps packages of other modules out of PYTHONPATH
module load python      # any Python >= 3.11, for example Python/3.12.3-GCCcore-13.3.0
python -m venv ~/envs/qorona && source ~/envs/qorona/bin/activate
pip install qorona
```

For the development version, see [Installation](installation.md).

## Batch jobs

| Flag | Use |
|------|-----|
| `--workers "$SLURM_CPUS_PER_TASK"` | Sets the number of threads to the cores of the job. Without it, numba uses all the cores it sees. |
| `--quiet` | Removes the progress bars, so the job logs stay readable. |

!!! note "One node, one task, many CPUs"
    Qorona uses shared-memory threads (numba), not MPI. Request the cores of one node with
    `--cpus-per-task`. Do not use `mpirun`.

On a GPU node, load the CUDA module of the site (for example `module load CUDA/12.4.0`), so that
numba finds the toolkit. `--device gpu` stops with an error if the GPU is not usable. See
[GPU acceleration](../gpu.md).

## SLURM templates

The two templates are in [`hpc/`](https://github.com/RayanDhib/Qorona/tree/main/hpc) in the
repository.

1. Replace the `<placeholders>` with the values of your site.
2. Keep the `--cluster` line only if one scheduler serves more than one cluster (for example VSC).
3. Set `--time` from the [GPU acceleration](../gpu.md) times.
4. If you use `--observer` or an export, add `--timestamp` and `--frame` to the build. See
   [Solution time and frame](../time-and-frame.md).

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
