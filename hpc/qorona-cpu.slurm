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
