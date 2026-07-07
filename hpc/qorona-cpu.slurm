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
