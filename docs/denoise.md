# Denoise full video runs

Use `scripts/denoise.py` to apply a trained model to full video runs.
The corrected NIfTI files are saved in BOLD units and follow the
source directory structure.

## Before you start

- Submit commands from the repository root.
- Use a checkpoint trained on the grade dataset.
- Check the data paths in `configs/paths/maguire.yaml`.
- Check that the chunk metadata and run-stats CSVs are configured in
  `configs/data/grade.yaml`.

The model architecture and chunk length are loaded from the checkpoint.

## Run denoising

Start with the first video run from each 2-month subject:

```bash
sbatch slurm/denoise.sbatch \
    checkpoint=runs/st_v4_ddp_disc_temporal_roi/best_model.pt \
    output_name=motion_corrected_st_v4 \
    runs=first_2mo
```

To process every video run from both ages, use `runs=all_video`.

| Selection | Runs included |
|---|---|
| `first_2mo` | First video run per 2-month subject; default selection for pipeline QC |
| `all_2mo` | All video runs from 2-month subjects; selection for ISC |
| `all_video` | All video runs from both ages |

### Check the configuration before submitting

Print the resolved settings and paths without starting denoising:

```bash
python scripts/denoise.py \
    checkpoint=runs/st_v4_ddp_disc_temporal_roi/best_model.pt \
    output_name=motion_corrected_st_v4 \
    runs=first_2mo \
    --cfg job --resolve
```

## Check the output

Results are written under
`<denoise_output_root>/<output_name>/`.

- Corrected runs end in `_corrected.nii.gz`.
- `denoise_log.csv` records each run's paths and status.
- Runs without normalisation statistics are marked
  `skipped_no_run_stats`.
- `manifest.json`, `config.yaml` and `run_info.jsonl` record the
  checkpoint, configuration and execution details.

The script refuses to use a non-empty output folder unless
`overwrite=true` is supplied.