#!/bin/bash
# Spatial smoothing of raw or denoised runs (CPU only; one ~450 MB run in memory at a time):
#   sbatch slurm/smooth.sbatch source=raw
#   sbatch slurm/smooth.sbatch source=motion_corrected_stv6_convlstm_all_2mo fwhm_mm=6
#SBATCH --job-name=moco_smooth
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
source ~/.bashrc  # before strict mode: /etc/bashrc reads unset variables
conda activate moco
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

python scripts/smooth.py "$@"
