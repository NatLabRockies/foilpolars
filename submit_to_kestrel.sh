#!/bin/bash
#SBATCH --job-name=foilpolars-full-sweep
#SBATCH --account=ai4wind
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --output=logs/full_sweep_%j.out
#SBATCH --error=logs/full_sweep_%j.err

set -euo pipefail

# Usage: ./submit_to_kestrel.sh [small|mhk] [nhours=N]
# nhours overrides the default 36h walltime and is consumed here, not
# passed on to run.sh
run_args=()
nhours=""
for arg in "$@"; do
    case "$arg" in
        nhours=*) nhours="${arg#nhours=}" ;;
        *) run_args+=("$arg") ;;
    esac
done

# Outside Slurm: resubmit ourselves via sbatch, applying --time if
# nhours was given, then stop here
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    sbatch_args=()
    if [[ -n "$nhours" ]]; then
        sbatch_args+=(--time="${nhours}:00:00")
    fi
    sbatch "${sbatch_args[@]}" "$0" "${run_args[@]}"
    exit 0
fi

# Avoid HDF5 file-locking errors on Lustre when checkpointing to netCDF
export HDF5_USE_FILE_LOCKING=FALSE

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Run the full pipeline via run.sh, forwarding the config tag (small,
# mhk, or none for configs/config.yaml)
./run.sh "${run_args[@]}"
