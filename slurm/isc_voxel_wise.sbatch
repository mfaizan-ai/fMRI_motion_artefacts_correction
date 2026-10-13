#!/bin/bash
# Voxel-wise LOSO ISC, raw vs a denoised source (CPU only; ~10 GB of voxel data per dataset, hence 64G):
#   sbatch slurm/isc_voxel_wise.sbatch source=motion_corrected_st_fc_beta_v1_all_2mo
#   sbatch slurm/isc_voxel_wise.sbatch source=motion_corrected_st_fc_beta_v1_all_2mo order=A
#SBATCH --job-name=moco_isc_voxel_wise
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
source ~/.bashrc  # before strict mode: /etc/bashrc reads unset variables
conda activate moco
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python scripts/isc_voxel_wise.py "$@"
