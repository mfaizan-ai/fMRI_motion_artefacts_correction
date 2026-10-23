# Spatially smooth runs

Use `scripts/smooth.py` to apply a 3D Gaussian kernel to raw or denoised video runs before
[voxel-wise ISC](isc_voxel_wise.md). Every volume is smoothed on its own, so time points are never mixed.
Smoothing raises voxel SNR and lets responses overlap across subjects despite residual misalignment
after normalisation; too wide a kernel blurs neighbouring regions together.

## Before you start

- Submit commands from the repository root.
- Smooth raw and denoised data with the **same** `fwhm_mm`. Otherwise part of a raw vs denoised ISC
  difference comes from the smoothing.
- Choose the FWHM without looking at the raw vs denoised result. The data are 2 mm isotropic, so the
  default 4 mm is two voxels; it is a starting point, not a validated optimum for infants.

## Run

Give each job a unique `--job-name` so its log (`logs/<job-name>_<job-id>.out`) can be found later, and
add a row to [slurm.md](slurm.md).

```bash
sbatch --job-name=smooth_raw_fwhm4 slurm/smooth.sbatch source=raw            # raw runs
sbatch --job-name=smooth_stv6_fwhm4 slurm/smooth.sbatch \
    source=motion_corrected_stv6_convlstm_all_2mo                            # a scripts/denoise.py output
sbatch --job-name=smooth_st_v4_fwhm4 slurm/smooth.sbatch source=motion_corrected_st_v4_all_video_runs \
    'denoised_root=${paths.legacy_denoised_root}'                            # original-repo runs
```

| Option | Default | Meaning |
|---|---|---|
| `source` | required | `raw`, or a denoised folder under `denoised_root` |
| `fwhm_mm` | `4` | kernel FWHM in mm |
| `preserve_zero_background` | `true` | voxels that are zero at every time point stay zero |
| `pattern` | `**/*task-videos*.nii.gz` | runs to smooth, as a glob under the input folder |
| `max_runs` | `null` | only the first N runs, for smoke tests |
| `overwrite` | `false` | allow writing into a non-empty output folder |

All 431 video runs took 2 h 15 min on 4 CPUs at 8 mm (about 19 s per run, 66 GB of output); one run (~450 MB) is
held in memory at a time.

`preserve_zero_background` keeps the brain-mask support, but the kernel is not renormalised at the mask
edge, so voxels just inside the edge are damped by the zero background.

### Check the configuration before submitting

```bash
python scripts/smooth.py source=raw fwhm_mm=8 --cfg job --resolve
```

## Output

Results are written under `<smoothed_root>/<source>_fwhm-<fwhm_mm>mm/` (`smoothed_root` is set in
`configs/paths/maguire.yaml`).

- The input directory tree and filenames are kept unchanged, so ISC finds the runs without any renaming.
- Runs are saved as float32 with the source affine and header.
- `smooth_log.csv` records each run's input and output path.
- `manifest.json` records the FWHM, run selection, nilearn version, git state and the source's own
  `manifest.json` (for denoised inputs). `config.yaml` and `run_info.jsonl` record the configuration and job.

## Voxel-wise ISC on smoothed data

Smooth raw and the denoised source at the same FWHM, then point ISC at the smoothed folders:

```bash
sbatch slurm/isc_voxel_wise.sbatch source=motion_corrected_stv6_convlstm_all_2mo_fwhm-4mm \
    'source_root=${paths.smoothed_root}/raw_fwhm-4mm' 'denoised_root=${paths.smoothed_root}'
```

Results go to `motion_denoising_qc/<source>_fwhm-4mm/isc_voxel_wise/`, separate from the unsmoothed results.
