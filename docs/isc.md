# Network ISC

Leave-one-out inter-subject correlation (ISC) measures how much of each brain region's signal is
driven by the video every infant watched. Good denoising should keep or raise it. ISC is computed
separately for each **video order** (A–F; each order is one video clip). The script works on raw runs
or on any denoised folder, at two levels: the 400 Schaefer ROIs and the 7 Yeo networks.

## Run

ISC needs **every** 2-month video run, not only the first. So denoise with `runs=all_2mo` first
(see [denoise.md](denoise.md)):

```bash
sbatch slurm/denoise.sbatch checkpoint=<path> output_name=motion_corrected_<name>_all_2mo runs=all_2mo
```

Then compute ISC on CPU. All orders take about 30–60 minutes, one order a few minutes:

```bash
sbatch slurm/isc_network.sbatch source=raw
sbatch slurm/isc_network.sbatch source=motion_corrected_<name>_all_2mo
sbatch slurm/isc_network.sbatch source=motion_corrected_<name>_all_2mo order=A      # one order only
sbatch slurm/isc_network.sbatch source=motion_corrected_st_v4_all_video_runs \
    'denoised_root=${paths.legacy_denoised_root}'                                  # original-repo runs
```

| Option | Default | Meaning |
|---|---|---|
| `source` | `raw` | `raw`, or a denoised folder under `denoised_root` |
| `order` | `null` | one order label (`A`–`F`), or every order in the segments CSV |
| `n_boot` / `ci` / `alpha` / `seed` | `10000` / `0.95` / `0.05` / `0` | bootstrap resamples, CI coverage, FDR level, RNG seed |
| `overwrite` | `false` | allow writing into an existing order folder |

## Method

This is a port of the original repo's `ISC_analysis/` (`build_isc_chunks.py`, `time_courses.py`,
`loo_isc.py`). There are two changes: segments are sliced straight from the runs instead of being
saved as intermediate NIfTI files, and significance comes from a bootstrap instead of a t-test.

1. **Segments.** `paths.isc_segments_csv` lists, for each subject, the 234-volume window of every
   viewing of each order's video. Its paths point at the uncropped derivative; they are re-rooted to
   the cropped, high-pass-filtered runs (`source_root`), or to the denoised copy of those runs.
   9-month visits (subject IDs ending in `A`) are excluded.
2. **Time series.** For each segment:
   - ROI signal: the mean over each ROI's voxels, using the 2-month Schaefer-400 atlas cropped like
     the runs.
   - Network signal: the unweighted mean of its ROIs' signals.
3. **Leave-one-out ISC.** Each subject's segment is correlated with the mean of the other subjects'
   segments. Each other subject contributes exactly one segment, never an average of its own
   viewings, chosen in this order: same session and run, same session, same run, nearest. A
   subject's viewings are averaged in Fisher-z. The network ISFC uses the same comparison but
   correlates every network with every other network.
4. **Significance.** Subject-wise bootstrap (Chen et al. 2016; BrainIAK `bootstrap_isc`):
   - Resample subjects with replacement `n_boot` times and take the mean Fisher-z ISC each time.
   - The 95% CI is read from those resamples (percentile method).
   - Two-sided p: the bootstrap distribution is shifted to centre on zero, and p is the share of
     shifted resamples at least as far from zero as the observed mean.
   - BH-FDR is applied across networks, and separately across ROIs.
   - The RNG is re-seeded for each order, so running one order alone gives the same numbers.

## Output

`motion_denoising_qc/<source>/isc_network/` (`original_data` for raw):

```
isc_network/
├── config.yaml, run_info.jsonl, isc_network.log
└── order_<A-F>/
    ├── manifest.json              # source, order, video, segments-CSV sha256, counts, bootstrap settings
    ├── segments.csv               # every segment used: subject, session, run, start/end volume, run path
    ├── network_isc_per_subject.csv, roi_isc_per_subject.csv     # subjects x networks / ROIs
    ├── network_isc_group.csv, roi_isc_group.csv                  # group_isc, ci_low, ci_high, p, q, significant
    ├── network_isfc_per_subject.npy                              # (S, 7, 7)
    ├── network_isfc_group.csv     # symmetric group ISFC; its diagonal is the network ISC
    ├── network_isc.png            # group ISC per network with bootstrap CI, * = FDR-significant
    └── network_isfc.png           # annotated 7x7 group ISFC
```

The old-vs-new check is in `runs/_equivalence/isc/`. It runs the original repo's own `loo_isc.py`
functions on its saved time courses for raw and st_v4, orders A–F, and compares the result with these
outputs.
