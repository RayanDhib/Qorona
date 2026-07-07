# HPC clusters

Qorona installs from PyPI into a plain virtual environment; no conda needed:

```bash
module purge            # optional; stops other modules leaking packages via PYTHONPATH
module load python      # any Python >= 3.11
python -m venv ~/envs/qorona && source ~/envs/qorona/bin/activate
pip install qorona
```

Two flags are made for batch jobs: `--workers` pins the kernel threads to your allocation
(numba otherwise takes every core it sees), and `--quiet` keeps job logs readable.

!!! note "One node, one task, many CPUs"
    Qorona is shared-memory threaded (numba), not MPI: request a full node's cores with
    `--cpus-per-task` and do not use `mpirun`.

On GPU nodes, load the site's CUDA module (`module load cuda`) so numba finds the toolkit;
`--device gpu` errors loudly if the GPU is unusable rather than silently falling back. See
[GPU acceleration](../gpu.md) for the backend details and timings.

## SLURM templates

Both templates live in
[`hpc/`](https://github.com/RayanDhib/Qorona/tree/main/hpc) in the repository;
replace the `<placeholders>` with your site's values.

### CPU node

```bash
#!/bin/bash -l
#SBATCH --cluster=<cluster>
#SBATCH --partition=batch
#SBATCH --account=<your-account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=5000MB
#SBATCH --job-name=qorona-cpu

#SBATCH -o qorona-cpu.stdout
#SBATCH -e qorona-cpu.stderr

# Qorona is shared-memory threaded (numba), not MPI: one node, one task, many
# CPUs. Request a full node's cores with --cpus-per-task and do NOT use mpirun.

# Load necessary modules
module purge
module load Python   # any Python >= 3.11; site module names vary

# Activate the environment Qorona is installed in.
# One-time setup:  python -m venv <path-to-venv>
#                  source <path-to-venv>/bin/activate
#                  pip install qorona
source <path-to-venv>/bin/activate

# Use exactly the cores SLURM gave us for the numba kernels.
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to the case directory
cd <case-directory>

# 1. Build the viewpoint-independent Q⊥ volume once (the minutes-scale,
#    parallel stage). This is what wants the full node.
qorona build solution.CFmesh -o solution.qor \
    --timestamp 2025-10-09T18:19:52 --outer-radius 8 \
    --device cpu \
    --workers $SLURM_CPUS_PER_TASK \
    --quiet

# 2. Render any number of viewpoints off that one volume (seconds each).
qorona render solution.qor -o eclipse.png --fov 8 --longitude 317 \
    --workers $SLURM_CPUS_PER_TASK --quiet
qorona render solution.qor -o polarity.png --fov 8 --longitude 317 --polarity-mode hue \
    --workers $SLURM_CPUS_PER_TASK --quiet
```

### GPU node

```bash
#!/bin/bash -l
#SBATCH --partition=<gpu-partition>
#SBATCH --account=<your-account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --job-name=qorona-gpu

#SBATCH -o qorona-gpu.stdout
#SBATCH -e qorona-gpu.stderr

# One GPU does the volume build; a handful of CPU cores feed it. Qorona is
# shared-memory threaded (numba), not MPI: one node, one task.

module purge
module load Python cuda   # site module names vary; any Python >= 3.11 plus a CUDA toolkit

# Activate the environment Qorona is installed in (see qorona-cpu.slurm for setup).
source <path-to-venv>/bin/activate

# Use exactly the cores SLURM gave us for the kernels that stay on the CPU.
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd <case-directory>

# Build on the GPU; --device gpu errors loudly if the GPU is unusable rather
# than silently falling back.
qorona build solution.CFmesh -o solution.qor \
    --timestamp 2025-10-09T18:19:52 --outer-radius 8 \
    --device gpu \
    --workers $SLURM_CPUS_PER_TASK \
    --quiet

# Renders are cheap; the same job can emit several viewpoints.
qorona render solution.qor -o eclipse.png --fov 8 --longitude 317 \
    --workers $SLURM_CPUS_PER_TASK --quiet
```
