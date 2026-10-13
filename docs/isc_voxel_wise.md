# Voxel-wise ISC

Voxel-wise leave-one-subject-out inter-subject correlation (ISC) maps, raw vs one denoising model, for each
video order. Where [network ISC](isc.md) gives one number per Yeo network, this script asks the same question
at every grey-matter voxel: where in the brain does the video drive a shared signal, and does denoising raise
or lower it? Raw and denoised data are analysed on the same subjects and voxels, so their difference is a
paired comparison.

## Run

The denoised source must hold every 2-month video run (`runs=all_2mo`, see [denoise.md](denoise.md)).
The job is CPU-only but memory-heavy: about 10 GB of voxel data per dataset, so the template asks for 64 GB.
One order takes about 30–60 minutes.

```bash
sbatch slurm/isc_voxel_wise.sbatch source=motion_corrected_<name>_all_2mo            # orders A-F
sbatch slurm/isc_voxel_wise.sbatch source=motion_corrected_<name>_all_2mo order=A    # one order
sbatch slurm/isc_voxel_wise.sbatch source=motion_corrected_st_v4_all_video_runs \
    'denoised_root=${paths.legacy_denoised_root}'                                     # original-repo runs
```

| Option | Default | Meaning |
|---|---|---|
| `source` | required | denoised folder under `denoised_root`; raw is always analysed alongside it |
| `order` | `null` | one order (`A`–`F`), or every order |
| `bootstrap.n_boot` / `seed` / `ci` | `1000` / `0` / `0.95` | subject bootstrap |
| `fdr_alpha` | `0.05` | BH-FDR level across voxels, per map |
| `overwrite` | `false` | allow writing into an existing order folder |

## Method

This is a port of the per-order core of the original repo's `ISC_analysis/isc_voxel_wise/` (`pipeline.py`,
`order_segments.py`, `loso_isc.py`, `bootstrap.py`, `figures.py`).

1. **Segments.** These are the same 234-volume video segments as network ISC, taken from
   `paths.isc_segments_csv`, with 9-month visits excluded. Each segment is paired with its raw run and its
   denoised run. Every run must be on one grid and affine, and long enough for its segment.
2. **Mask.** The 2-month grey-matter template mask is resampled (nearest) onto the data grid. The script
   stops if the data grid cuts off part of the mask.
3. **Leave-one-subject-out ISC.** Each segment's voxel time series is z-scored. A subject's representative
   is the mean of its z-scored scans. A subject's reference is the equal-weight mean of every *other*
   subject's representative. Each of the subject's own scans is correlated with the reference voxel by
   voxel, and the scan maps are averaged in Fisher z. The result is one map per subject.
4. **Common mask.** Only voxels with signal in every scan of both datasets are kept, so all maps cover the
   same voxels. `corrected_minus_raw` is the per-subject paired difference.
5. **Statistics.** Subject bootstrap of the group mean Fisher z:
   - The same resamples are used for raw, corrected and their difference, so the difference test is
     paired.
   - The CI is the percentile interval of the bootstrap means.
   - p is two-sided and sign-based: how often the bootstrap means fall on the far side of 0. The smallest
     possible p is 2 / (n_boot + 1).
   - BH-FDR is applied across voxels, separately for each map.
6. **Networks.** Each subject's map is averaged within each Yeo-7 network (Schaefer-400 resampled onto the
   grid), then bootstrapped across subjects with the same resamples.

Caveat: leave-one-out maps are not independent across subjects, since every reference contains the other
subjects. The bootstrap treats subjects as independent, so CIs are likely somewhat narrow.

## Output

`motion_denoising_qc/<source>/isc_voxel_wise/order_<A-F>/`:

```
order_A/
├── manifest.json                # source, order, video, segments-CSV sha256, counts, bootstrap settings
├── common_mask.nii.gz           # voxels every map covers
├── subject_maps/
│   ├── subjects.csv             # volume index -> subject
│   └── subject_isc_z_{raw,corrected}.nii.gz      # 4D, one Fisher-z map per subject
├── group_maps/{raw,corrected,corrected_minus_raw}_{mean_z,ci_low_z,ci_high_z,q_fdr,mean_z_fdr}.nii.gz
├── tables/
│   ├── segments.csv, scan_qc.csv               # segments used; per-scan median r
│   ├── subject_summary.csv                     # scans per subject, median z per map
│   ├── group_summary.csv                       # per map: median ISC, % voxels CI > 0 / < 0, FDR counts
│   └── network_isc.csv                         # per network and map: mean z with bootstrap CI
└── figures/
    ├── group_maps.png, group_maps_fdr.png      # axial slices: raw, corrected, corrected - raw
    ├── peak_views.png                          # three views through the raw ISC peak
    ├── isc_distributions.png                   # voxel histograms; each infant's median raw vs corrected
    ├── network_isc.png                         # network ISC and the paired difference
    └── interactive/*.html                      # clickable viewers
```

`mean_z_fdr` holds the group mean where the FDR test is significant and NaN elsewhere. Run settings are in
`<source>/isc_voxel_wise/{config.yaml,run_info.jsonl,isc_voxel_wise.log}`.

## Example

```bash
sbatch --job-name=isc_voxel_wise_st_fc_beta_v1_A slurm/isc_voxel_wise.sbatch \
    source=motion_corrected_st_fc_beta_v1_all_2mo order=A
```

This writes `motion_denoising_qc/motion_corrected_st_fc_beta_v1_all_2mo/isc_voxel_wise/order_A/`. Start
with `tables/group_summary.csv`:
- the `corrected_minus_raw` row's `median_mean_z` gives the typical voxel change
- its `n_fdr_positive` and `n_fdr_negative` count the voxels where denoising raised or lowered ISC
  significantly

Then look at `figures/group_maps_fdr.png` for where those voxels are.

The old-vs-new check is in `runs/_equivalence/isc_voxel_wise/`. It compares the st_v4, order A outputs map
by map with the original repo's.
