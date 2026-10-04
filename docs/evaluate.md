# Evaluating a checkpoint on the test split

`scripts/evaluate.py` corrects every held-out test chunk and reports fMRI quality metrics by motion grade.
`scripts/visualization/plot_test.py` turns those metrics into box plots.

## Test metrics

Only video chunks of 2-month subjects from the frozen test split are used (9-month visits, IDs ending in
"A", are excluded). Each 5-volume chunk is normalised with its run's stats, corrected by the generator and
denormalised inside the input brain mask. Input and corrected chunks are then compared:

| Metric | Improvement | Positive means |
|---|---|---|
| DVARS | raw − corrected | less frame-to-frame change |
| tSNR | corrected − raw | higher temporal SNR |
| Global signal std | raw − corrected | less global fluctuation |
| Spatial smoothness | corrected / raw (ratio) | > 1: larger spatial gradients, i.e. less smooth |

```bash
sbatch slurm/evaluate.sbatch checkpoint=<path to .pt> output_dir=<run dir>/test
```

`checkpoint` must be the `.pt` file, not the run folder. Write `output_dir` into a subfolder such as
`test/`, because the script saves `config.yaml` there and would overwrite a run folder's own config.

Outputs in `output_dir`:

- `test_metrics_per_chunk.csv`: one row per chunk with every input, corrected and improvement value
- `test_metrics_summary_by_grade.csv`: mean and std of the improvements per grade
- `evaluate.log`, `config.yaml`, `run_info.jsonl`

## Plots

```bash
python scripts/visualization/plot_test.py input_dir=<evaluate output_dir>
```

This runs on CPU in a few seconds, so it doesn't need SLURM. It reads `test_metrics_per_chunk.csv` and
writes `plots/test_metrics_by_grade.png` into `input_dir`, plus `config.yaml` and `run_info.jsonl`.

The figure has six panels. The top row shows tSNR, DVARS and global signal std for raw (orange) and
corrected (blue) chunks at each grade. The bottom row shows the improvement of each metric, with boxes
shaded by grade. Boxes show quartiles, with whiskers at 1.5 IQR. The lines join the grade medians, with a
95% percentile-bootstrap CI.

| Option | Default | Meaning |
|---|---|---|
| `input_dir` | required | folder written by `evaluate.py` |
| `output_dir` | `<input_dir>/plots` | where the figure is saved |
| `n_boot`, `ci`, `seed` | `2000`, `0.95`, `0` | bootstrap settings for the median CI |

Colours, labels and sizes are in `configs/plot_test.yaml`.
