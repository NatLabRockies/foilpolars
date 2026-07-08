#!/bin/bash
#SBATCH --job-name=foilpolars-full-sweep
#SBATCH --account=ai4wind
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --output=logs/full_sweep_%j.out
#SBATCH --error=logs/full_sweep_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Start from a clean output dir, then rebuild the Grassmann basis,
# run the full sweep, and regenerate the comparison figures
rm -rf output
uv run foilpolars grassmann --config configs/sweep_config.yaml
uv run foilpolars run --config configs/sweep_config.yaml
uv run foilpolars plot --config configs/sweep_config.yaml
