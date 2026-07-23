#!/bin/bash
#SBATCH --job-name=foilpolars-full-sweep
#SBATCH --account=ai4wind
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --output=logs/full_sweep_%j.out
#SBATCH --error=logs/full_sweep_%j.err

set -euo pipefail

# Avoid HDF5 file-locking errors on Lustre when checkpointing to netCDF
export HDF5_USE_FILE_LOCKING=FALSE

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Rebuild the Grassmann basis, run the full sweep, and regenerate the
# comparison figures
uv run foilpolars grassmann --config configs/sweep_config.yaml
uv run foilpolars run --config configs/sweep_config.yaml
uv run foilpolars plot --config configs/sweep_config.yaml
