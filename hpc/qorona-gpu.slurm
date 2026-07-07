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
