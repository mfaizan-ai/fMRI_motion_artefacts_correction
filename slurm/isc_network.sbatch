#!/bin/bash
# Leave-one-out network ISC (CPU only):
#   sbatch slurm/isc_network.sbatch source=raw
#   sbatch slurm/isc_network.sbatch source=motion_corrected_st_fc_beta_v1_all_2mo order=A
#SBATCH --job-name=moco_isc_network
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
source ~/.bashrc  # before strict mode: /etc/bashrc reads unset variables
conda activate moco
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python scripts/isc_network.py "$@"
