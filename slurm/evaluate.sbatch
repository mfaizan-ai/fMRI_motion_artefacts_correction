#!/bin/bash
# Test-split metrics:  sbatch slurm/evaluate.sbatch checkpoint=runs/st_v4/best_model.pt output_dir=runs/st_v4/test
#SBATCH --job-name=moco_evaluate
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
set -euo pipefail
source ~/.bashrc
conda activate moco
cd "$SLURM_SUBMIT_DIR"
python scripts/evaluate.py "$@"
