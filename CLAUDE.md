# fmri-moco

Disentangled 3D CycleGAN for motion correction of infant fMRI (FoundCog).In this repo, we are correcting the motion with the CycleGANS model with unpaired corrupted and corrected data. So the motion data is chunk size of 5 and the we have different motion grades for instance, grade 1 (very low motion), grade 2 (low motion) to grade 6 (extreme motion). for this we are deveoping a cycleGNAS architecure and the pipline to correct for th head motion artefacts in the infatns fMRI data. Futhermore, you know the whole details in this repo. 

PyTorch, Hydra configs, trained with SLURM on Maguire (H200, data on Lustre).

## Layout

- `src/moco/` – all logic (data, models, losses, training, evaluation, utils). Installed with `pip install -e .`
- `scripts/` – thin entry points only: load config, call functions from `src/moco`. useful minimal code and remove things tah are not needed to avoid writing big complex scripts. 
- `configs/` – Hydra YAML. Every hyperparameter and path lives here.
- `configs/paths/*.yaml` – the only place absolute paths may appear.
- `slurm/` – sbatch templates that call scripts with Hydra overrides.
- `splits/` – frozen subject-level split CSVs. Never re-split at train time.
- `notebooks/` – exploration only. Nothing in `src/` imports from here.

Or organize based on the existing content as you find it useful to organize that work best for this project. Dont create mess, i dont like mess and i want you to make my work clean and reporduecilbe and more reproducible than now I have. 

## Rules
- No hardcoded paths, hyperparameters, or magic numbers in code. Read them from `cfg`. If a new setting is needed, add it to the relevant YAML with a sensible default.
- One responsibility per module. Don't put data loading, model, training loop and plotting in one file.
- Write the simplest code that does the job. No speculative abstractions, base classes, factories, or options nobody asked for.
- Don't add new dependencies without asking.
- Reuse existing functions in `src/moco` before writing new ones. Search first.
- Prefer editing existing files over creating new ones.

## Style
- Code should read like an experienced researcher wrote it, not generated boilerplate.
- No long docstrings or banner comments at the top of files. One line saying what the module is for is enough.
- Comment only non-obvious logic (why, not what): shape conventions, numerical tricks, neuroimaging-specific choices. One short line, not paragraphs. Never comment every line.
- Short docstrings on public functions; include tensor shapes, e.g. `x: (B, T, X, Y, Z)` with time as channels.
- Type hints on function signatures.
- Clear names over comments. `fd_mean`, not `x2`.
- No print debugging left behind; use `logging`.
- Format and lint with ruff.

## Reproducibility

- Seed everything from `cfg.seed` (python, numpy, torch, cuda, dataloader workers).
- Each run saves its resolved config, git commit + dirty flag, and SLURM job ID into its output dir.
- Derived data (chunks) is written with a `manifest.json` describing how it was made.

## Tests

- Small pytest tests for things that silently break: output shapes, overlap-add stitching, FD categorisation boundaries, history buffer.
- Run `pytest` and `ruff check` after changes.

## Working with me

- For changes touching more than a couple of files, briefly state the plan first.
- Keep explanations in chat short. Don't summarise every edit.
- If something is ambiguous (e.g. which loss weight, which split), ask rather than guess.

## Note
please dont change this repo, and i would recommend you instead create another directory and make the strucure there, I would like to keep the original repo struucre unchanged. instad i would recommend you to put all the changes you make to this repo in this directory. /lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion_aretefacts_correction_cycle_gans_updated, so make sure you create the repo and strucure of the project by following hte best practices, the pratices that are not even mentiond here. And copy this claude.md there too. 