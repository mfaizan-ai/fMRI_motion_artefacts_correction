#!/bin/bash
# 4-GPU DDP training. Any Hydra override can be appended:
#   sbatch slurm/train.sbatch experiment=stv6_convlstm
#   sbatch slurm/train.sbatch experiment=st_v4 run_name=st_v4_300ep train.epochs=300
#SBATCH --job-name=moco_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
set -euo pipefail
source ~/.bashrc
conda activate moco
cd "$SLURM_SUBMIT_DIR"
torchrun --standalone --nproc_per_node=4 scripts/train.py "$@"
