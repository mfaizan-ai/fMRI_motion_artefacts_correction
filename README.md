# moco: motion correction of infant fMRI with a disentangled CycleGAN

Unpaired CycleGAN that maps motion-corrupted 5-volume chunks (Grades 2-6) to the motion-free
domain (Grade 1), for FoundCog infant fMRI. Cleaned-up, config-driven version of
`../motion-artefacts-correction`; that repo is left unchanged.

## Setup

```bash
conda activate moco
pip install -e ".[dev]"
```

## Layout

```
configs/           Hydra configs; configs/paths/*.yaml is the only place with absolute paths
  train.yaml       entry config for training (data / model / train / atlas groups)
  experiment/      named experiments (st_v4, stv6_convlstm)
  denoise.yaml     whole-run correction
  evaluate.yaml    test-split metrics
src/moco/
  data/            grade dataset (robust normalisation), legacy PSC + sequence datasets, loaders
  models/          spatiotemporal (factorized R(3+1)D, optional ConvLSTM) and legacy disentangled CycleGANs
  losses/          GAN / cycle / identity, ROI-timeseries, sequence (temporal, FC)
  training/        epoch loop, trainer, schedules, checkpoints, DDP
  evaluation/      fMRI metrics, whole-run denoising, test-set evaluation
  atlas.py         age-appropriate Schaefer-400 atlases and ROI extraction
scripts/           thin entry points
slurm/             sbatch templates (Hydra overrides are passed through)
splits/            frozen split assignments + manifest
tests/             pytest
```

## Usage

```bash
# training (4-GPU DDP); outputs go to runs/<run_name>/
sbatch slurm/train.sbatch experiment=st_v4
sbatch slurm/train.sbatch experiment=stv6_convlstm
sbatch slurm/train.sbatch experiment=st_v4 run_name=st_v4_300ep train.epochs=300

# quick smoke test on one GPU
python scripts/train.py experiment=st_v4 run_name=smoke train.epochs=1 \
    train.max_train_batches=5 train.max_val_batches=2 train.val_every=1 wandb.enabled=false

# inspect the composed config without running
python scripts/train.py experiment=stv6_convlstm --cfg job --resolve

# whole-run denoising and test metrics (any checkpoint, including the original repo's)
sbatch slurm/denoise.sbatch checkpoint=runs/st_v4/best_model.pt output_name=motion_corrected_st_v4 runs=all_video
sbatch slurm/evaluate.sbatch checkpoint=runs/st_v4/best_model.pt output_dir=runs/st_v4/test
```

Each run directory holds the resolved `config.yaml`, `run_info.jsonl` (git commit, dirty flag,
SLURM job ID, host, one line per (re)start), `train.log`, per-epoch CSVs and checkpoints.
`latest.pt` is resumed automatically when a job is resubmitted with the same `run_name`.

## Reproducibility

- Everything is seeded from `seed` (per-rank offset under DDP); `deterministic=true` also makes cuDNN
  deterministic. GPU training is still not bit-reproducible: the trilinear-upsample backward and the
  `index_add` in ROI extraction have no deterministic CUDA kernels, so two identical runs drift slightly
  (~1e-3 in losses after a few steps; the original code behaves the same).
- Checkpoints store the resolved config; `moco.models.build.load_model` also reads checkpoints from
  the original repo (their `args`), giving identical outputs.
- Splits are read from `splits/`, never recomputed at train time.

## Known issue: the legacy split is not subject-level

`splits/grade_split_legacy.csv` reproduces the original runs exactly but assigns splits per
(subject, task): 64 of 180 subjects have, for example, rest chunks in train and video chunks in
test, and 2-month (`ICC105`) and 9-month (`ICC105A`) visits of the same infant can land in
different splits. A subject-level split (by infant, both visits together) should be frozen as a
new file before reporting held-out results.

## Checks

```bash
pytest
ruff check .
```
