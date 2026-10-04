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

## Splits

- `splits/grade_subject_split.csv` (default): one row per `subject_id`, 70/15/15, stratified by age group
  and by each subject's share of Grade 4-6 chunks. Made by `python scripts/make_splits.py`; the manifest
  records the source CSV and its SHA-256, seed, ratios and per-split counts. A different dataset gets its
  own split: `python scripts/make_splits.py name=<new> chunk_metadata_csv=<path>`. Existing files are never
  overwritten unless `overwrite=true`.
- `splits/grade_split_legacy.csv`: the original per (subject, task) assignment, pinned by
  `experiment=st_v4` / `stv6_convlstm` so those runs reproduce. 64 subjects have chunks in more than one
  split there, so it should not be used for new held-out results.
- The unit is the subject ID, so the 2- and 9-month visits of one infant (e.g. ICC105 / ICC105A) can
  fall in different splits.

## Checks

```bash
pytest
ruff check .
```
