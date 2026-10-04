#!/bin/bash
# Whole-run correction:
#   sbatch slurm/denoise.sbatch checkpoint=runs/st_v4/best_model.pt output_name=motion_corrected_st_v4 runs=all_video
#SBATCH --job-name=moco_denoise_stv6_convlstm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
source ~/.bashrc  # before strict mode: /etc/bashrc reads unset variables
conda activate moco
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python scripts/denoise.py "$@"
