#!/bin/bash
# Pipeline-level QC (CPU only):
#   sbatch slurm/pipeline_qc.sbatch source=raw
#   sbatch slurm/pipeline_qc.sbatch source=motion_corrected_stv6_convlstm_first_run_video_data
#SBATCH --job-name=moco_pipeline_qc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
source ~/.bashrc  # before strict mode: /etc/bashrc reads unset variables
conda activate moco
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python scripts/pipeline_qc.py "$@"
