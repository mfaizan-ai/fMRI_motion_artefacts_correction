#!/bin/bash
#SBATCH --job-name=moco_equiv
#SBATCH --output=/lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion_aretefacts_correction_cycle_gans_updated/runs/_equivalence/job_%j.out
#SBATCH --error=/lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion_aretefacts_correction_cycle_gans_updated/runs/_equivalence/job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
source ~/.bashrc
conda activate moco
export WANDB_MODE=disabled
SMOKE="--epochs 1 --batch_size 4 --num_workers 2 --max_train_batches 6 --max_val_batches 3 --val_every 1 --save_every 1"
echo "=== old"
cd /lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion-artefacts-correction && python train.py --use_grade_dataset \
  --grade_chunk_metadata_csv /lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/pytorch-CycleGAN-and-pix2pix/data_preprocessing/motion_grades_chunk_5_dataset_hfiltered/chunk_metadata.csv \
  --grade_run_stats_csv /lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion-artefacts-correction/all_dataset/run_normalization_stats_hfiltered.csv \
  --in_timepoints 5 $SMOKE --run_name old --ckpt_root /lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion_aretefacts_correction_cycle_gans_updated/runs/_equivalence \
  --max_grad_norm 3.0 --w_cyc 10.0 --w_idt 5.0 --d_update_every 1 --label_smooth_real 1.0 --label_smooth_fake 0.0 \
  --r1_weight 0.5 --r1_every 8 --num_disc_scales 2 --residual --use_st_model \
  --use_roi_discriminator --lambda_roi 0.5 --w_roi_adv 1.0 --w_roi_cycle 1.0 --deterministic --no_wandb
echo "=== new"
cd /lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion_aretefacts_correction_cycle_gans_updated && python scripts/train.py experiment=st_v4 run_name=_equivalence/new paths.runs_root=/lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion_aretefacts_correction_cycle_gans_updated/runs \
  train.epochs=1 train.num_workers=2 train.max_train_batches=6 train.max_val_batches=3 train.val_every=1 \
  train.save_every=1 deterministic=true wandb.enabled=false
