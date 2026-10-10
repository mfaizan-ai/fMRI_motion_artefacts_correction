# Pipeline-level QC of raw and denoised runs

`scripts/pipeline_qc.py` measures how much motion is left in a set of whole runs, and whether real
connectivity survives. It runs on one data source at a time: the original runs, or the output of one
model from `scripts/denoise.py`. Results go to `motion_denoising_qc/` in the repo root, which git
ignores:

```
motion_denoising_qc/
  original_data/                                        # source=raw
  motion_corrected_stv6_convlstm_first_run_video_data/  # source=<denoised output_name>
```

## Runs

The script uses the first video run of each 2-month subject (`select_runs(..., "first_2mo")`, 128
subjects). A denoised source must contain every one of these runs, found by the naming rule both repos
use: the same relative folder, with a `_corrected.nii.gz` suffix.
That way raw data and every model are compared on the same subjects, and the script stops if a run is
missing.

## Metrics

Per subject, written to `per_subject.csv`:

| Column | Definition |
|---|---|
| `mean_fd` | mean framewise displacement (Power 2012) of the raw run |
| `Q` | signed asymmetric modularity of the consensus Louvain partition of the subject's FC (netneurotools `negative_asym`, 100 repeats, γ = 1) |
| `n_communities` | number of communities in that partition |
| `dvars` | mean non-standardised DVARS, nipype convention (brain median scaled to 1000) |
| `tsnr` | mean voxel tSNR in the brain, nipype convention |

FC uses Schaefer-400 ROI means (cropped 2-month atlas), with a 0.01 Hz cosine high-pass, Pearson
correlation and a Fisher z-transform. Mean FD always comes from the raw run, since denoising does not
change head motion.

Group level, written to `summary.json` and the `.npy` matrices (400×400, NaN diagonal):

| Output | Definition |
|---|---|
| `qc_fc_r.npy`, `qc_fc_p.npy` | QC-FC: correlation of each edge's FC with mean FD across subjects |
| `qc_fc_fdr_significant.npy` | QC-FC edges significant after Benjamini–Hochberg FDR (α = 0.05) |
| `qcfc_median_abs_r` | median \|QC-FC r\| over all edges |
| `qcfc_dd_r`, `qcfc_dd_p` | QC-FC-DD: correlation of QC-FC r with the distance between ROI centres (`distance_matrix.npy`) |
| `fc_mean_z.npy`, `fc_p.npy`, `fc_q.npy`, `fc_significant.npy` | FC significance: one-sample t-test of each edge against 0, FDR-corrected |
| `Q_mean`, `Q_sd`, `Q_vs_fd_r`, `Q_vs_fd_p` | modularity summary, and its correlation with mean FD |

Lower QC-FC (fewer significant edges, smaller median \|r\|) and a QC-FC-DD closer to 0 mean less
motion is left. FC significance and Q show whether real network structure survived denoising.

Each folder also holds `roi_timeseries/` (one `(T, 400)` array per subject), `pipeline_qc.log`,
`config.yaml` and `run_info.jsonl`.

## Usage

```bash
sbatch slurm/pipeline_qc.sbatch source=raw
sbatch slurm/pipeline_qc.sbatch source=motion_corrected_stv6_convlstm_first_run_video_data
```

The job needs only CPUs; modularity takes most of the time. Settings are in `configs/pipeline_qc.yaml`.

| Option | Default | Meaning |
|---|---|---|
| `source` | `raw` | `raw`, or a folder under `paths.denoise_output_root` |
| `overwrite` | `false` | the script exits if the output folder already has files, unless this is `true` |
| `max_subjects` | `null` | use only the first N subjects, for smoke tests |
| `edge_alpha` | `0.05` | FDR level |
| `modularity.repeats` / `gamma` / `seed` | `100` / `1.0` / `12345` | consensus Louvain settings |

This is a port of `motion_denoising_measure/denoising_evaluation.py` from the original repo. The
old-vs-new comparison is in `runs/_equivalence/pipeline_qc/`.

## Figures
```bash
python scripts/visualization/plot_pipeline_qc.py
python scripts/visualization/plot_pipeline_qc.py comparison_name=raw_vs_stv6 \
    'sources=[original_data,motion_corrected_stv6_convlstm_first_run_video_data]' 'labels=[Raw,stv6]'
```

These figures follow the style of the original repo's `plot_denoising_qc.py`, with one panel or row
per source. `sources` lists folders under
`motion_denoising_qc/`, `labels` gives one plot label per source and `colors` one colour per source. All sources must have been
computed on the same subjects. The script runs on CPU in under a minute and writes to
`motion_denoising_qc/figures/<comparison_name>/`:

| File | Shows |
|---|---|
| `qc_fc_distribution.png` | Ciric-style QC-FC density, one panel per source, zero line clipped at the curve, with median \|QC-FC\| |
| `qc_fc_matrix.png` | 400×400 QC-FC r matrices on a shared ±1 scale |
| `fc_matrix.png` | 400×400 group-mean Fisher-z FC matrices on a shared symmetric scale |
| `qc_fc_distance_dependence.png` | QC-FC r against ROI distance: density contours, zero line, red linear fit, QC-FC-DD r |
| `qc_fc_significant_edges.png` | sagittal glass brain per source of every FDR-significant QC-FC edge, coloured by \|r\| |
| `modularity_q.png` | box plot of per-subject Q, one colour per source (no outlier points) |
| `fc_significant_edges.png` | sagittal glass brain per source of the strongest `fc_top_percent` % (default 2) of significant FC edges, signed mean Fisher-z, with the \|z\| threshold |

Every figure has one panel per source in a single row. Matrix and glass-brain colour bars are drawn as
tall as the panels. The default compares Raw, stv4, stv6_convlstm and st_fc_beta_v1.

The original repo drew the distance-dependence density with seaborn. Here it is a smoothed 2-D histogram
with the same contour levels, which looks the same without the extra dependency. Glass-brain node
positions come from the adult MNI Schaefer-400 centroid table (`paths.schaefer_mni_centroids_csv`).
These plots are schematic, and the ROI order matches the infant atlas used for the metrics.

Runs denoised by the original repo (for example `motion_corrected_st_v4`) live outside
`paths.denoise_output_root`. Compute their QC with
`sbatch slurm/pipeline_qc.sbatch source=motion_corrected_st_v4 'denoised_root=${paths.legacy_denoised_root}'`.
