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
