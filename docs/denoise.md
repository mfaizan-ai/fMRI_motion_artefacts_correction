# Denoising whole runs with trained models

`scripts/denoise.py` uses a trained checkpoint to correct full video runs. It writes the corrected runs
in BOLD units, keeping the same directory tree as the source data.

## How it works

For each selected run:

1. Loads the preprocessed volume (brain-masked, cropped to 60×72×56, high-pass filtered, common space).
2. Normalises it with the run's robust stats, `(x - median) / scale`, the same normalisation used in
   training. Background stays 0.
3. Splits the run into non-overlapping chunks (0–4, 5–9, ...), using the chunk size stored in the
   checkpoint. It pads axis 0 to 64, corrects each chunk with the generator, then crops back. A short
   final chunk is filled by repeating its last volume, and the filler is dropped afterwards.
4. Concatenates the chunks back into the full run, denormalises it inside the brain mask, and saves it
   with the source affine and header.

The model is rebuilt from the config stored in the checkpoint. 

Only checkpoints trained on the grade dataset are valid inputs, for example `st_v4_ddp_disc_temporal_roi`,
`stv6_convlstm_cycleGANS`, `st_fc_beta_v1` and `st_v3_ddp_with_roi_time_series_cycleGANS` in `runs/`.

## Usage

```bash
sbatch slurm/denoise.sbatch checkpoint=<path> output_name=<folder> [runs=first_2mo|all_video] [overwrite=true]
```

Submit from the repo root. The sbatch file only requests a GPU, activates the `moco` environment and
passes the overrides on to `scripts/denoise.py`. Every other setting comes from `configs/denoise.yaml`.

| Option | Default | Meaning |
|---|---|---|
| `checkpoint` | required | path to a `.pt` checkpoint |
| `output_name` | required | output folder, created under `paths.denoise_output_root` |
| `runs` | `first_2mo` | `first_2mo`: first video run of each 2-month subject. `all_video`: every video run, both ages |
| `overwrite` | `false` | the script exits if the output folder already has files, unless this is `true` |
| `source_root` | `paths.denoise_source_root` | root of the input runs |
| `output_root` | `<denoise_output_root>/<output_name>` | override to write somewhere else |
| `device` | `cuda` | |

Absolute paths live in `configs/paths/maguire.yaml`. The list of runs comes from the chunk metadata
CSV, and the normalisation stats come from the run-stats CSV; both are set in `configs/data/grade.yaml`.

## Examples

```bash
# all video runs with st_v4
sbatch slurm/denoise.sbatch checkpoint=runs/st_v4_ddp_disc_temporal_roi/best_model.pt \
    output_name=motion_corrected_st_v4_v2 runs=all_video

# first 2-month run per subject with the ConvLSTM model
sbatch slurm/denoise.sbatch checkpoint=runs/stv6_convlstm_cycleGANS/best_model.pt \
    output_name=motion_corrected_stv6_convlstm

# print the resolved config (all paths) without running anything
python scripts/denoise.py checkpoint=runs/st_v4_ddp_disc_temporal_roi/best_model.pt \
    output_name=test --cfg job --resolve
```

## Output

```
<denoise_output_root>/<output_name>/
├── _subject_id_<ID>/_referencetype_standard/_run_<NNN>_session_<N>_task_name_videos/
│   └── <source name>_corrected.nii.gz    # same tree as source_root
├── manifest.json      # checkpoint path + sha256 + epoch, chunk length, run selection, input CSVs, git state
├── config.yaml        # resolved config
├── run_info.jsonl     # git commit, dirty flag, SLURM job ID, host (one line per start)
├── denoise.log
└── denoise_log.csv    # one row per run: paths, status (ok / skipped_no_run_stats), checkpoint, git commit
```

Runs with no entry in the run-stats CSV are skipped and logged. SLURM stdout and stderr go to
`logs/moco_denoise_<jobid>.out` and `.err`.
