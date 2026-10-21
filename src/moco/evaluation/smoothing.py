"""Spatial Gaussian smoothing of whole raw or denoised runs, mirroring the input tree, as input to voxel-wise ISC."""
import json
import logging
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import smooth_img
from omegaconf import DictConfig

from moco.evaluation.pipeline_qc import RAW_SOURCE
from moco.utils.run_info import git_state

log = logging.getLogger(__name__)


def input_dir(cfg: DictConfig) -> Path:
    """`source_root` for raw runs, `denoised_root/<source>` for a scripts/denoise.py output."""
    return Path(cfg.source_root) if cfg.source == RAW_SOURCE else Path(cfg.denoised_root) / cfg.source


def find_runs(root: Path, pattern: str) -> list[Path]:
    """Runs under `root` matching the glob `pattern`.

    Args:
        root: Input folder (raw source root or one denoised folder).
        pattern: Glob relative to `root`, e.g. "**/*task-videos*.nii.gz".

    Returns:
        Matching run paths, sorted so `max_runs` always picks the same runs.

    Raises:
        FileNotFoundError: If nothing matches.
    """
    runs = sorted(root.glob(pattern))
    if not runs:
        raise FileNotFoundError(f"no runs match {pattern!r} under {root}")
    return runs


def smooth_run(volume: np.ndarray, affine: np.ndarray, fwhm_mm: float, preserve_zero_background: bool) -> np.ndarray:
    """Gaussian-smooth every volume of a run in space only; time points are never mixed.

    Args:
        volume: (X, Y, Z, T) run.
        affine: Voxel-to-mm affine; the FWHM is converted to voxels from it.
        fwhm_mm: Kernel full width at half maximum, in mm.
        preserve_zero_background: Reset voxels that are zero at every time point (outside the brain mask) to zero.
            The kernel is not renormalised at the mask edge, so boundary voxels are still damped by the background.

    Returns:
        (X, Y, Z, T) float32 smoothed run.

    Raises:
        ValueError: If the run is not 4D or holds NaN/Inf.
    """
    if volume.ndim != 4:
        raise ValueError(f"expected a 4D run, got shape {volume.shape}")
    if not np.isfinite(volume).all():
        raise ValueError("run contains NaN/Inf")
    smoothed = smooth_img(nib.Nifti1Image(volume, affine), fwhm=fwhm_mm).get_fdata(dtype=np.float32)
    if preserve_zero_background:
        smoothed[~np.any(volume != 0, axis=3)] = 0
    return smoothed


def write_manifest(output_root: Path, source_dir: Path, runs: list[Path], cfg: DictConfig) -> None:
    """Describe how the smoothed folder was made in output_root/manifest.json.

    Args:
        output_root: Smoothing output folder.
        source_dir: Folder the runs were read from; its own manifest.json (denoised sources) is linked.
        runs: Selected input runs.
        cfg: Composed `configs/smooth.yaml`.
    """
    source_manifest = source_dir / "manifest.json"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": cfg.source,
        "source_dir": str(source_dir),
        "source_manifest": str(source_manifest) if source_manifest.exists() else None,
        "fwhm_mm": float(cfg.fwhm_mm),
        "preserve_zero_background": cfg.preserve_zero_background,
        "pattern": cfg.pattern,
        "n_runs": len(runs),
        "nilearn": version("nilearn"),
        **git_state(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run_smoothing(cfg: DictConfig) -> None:
    """Smooth every selected run and save it under output_root with the same relative path and filename.

    Filenames are kept so scripts/isc_voxel_wise.py reads smoothed data through its root overrides alone.

    Args:
        cfg: Composed `configs/smooth.yaml`; uses source, source_root, denoised_root, output_root, pattern,
            fwhm_mm, preserve_zero_background and max_runs.

    Raises:
        ValueError: If a run's spatial units are set and not mm, since the FWHM is given in mm.
    """
    source_dir, output_root = input_dir(cfg), Path(cfg.output_root)
    runs = find_runs(source_dir, cfg.pattern)[: cfg.max_runs]
    log.info("smoothing %d runs from %s at FWHM %g mm", len(runs), source_dir, cfg.fwhm_mm)
    write_manifest(output_root, source_dir, runs, cfg)
    log_path = output_root / "smooth_log.csv"

    for i, source_path in enumerate(runs, 1):
        img = nib.load(source_path)
        # 330/431 source video runs leave the units code unset; their affine is still in mm (2 mm grid)
        if img.header.get_xyzt_units()[0] not in ("mm", "unknown"):
            raise ValueError(f"{source_path}: spatial units are {img.header.get_xyzt_units()[0]!r}, not mm")
        smoothed = smooth_run(np.asarray(img.dataobj, dtype=np.float32), img.affine, float(cfg.fwhm_mm),
                              cfg.preserve_zero_background)
        out_path = output_root / source_path.relative_to(source_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        header = img.header.copy()
        header.set_data_dtype(np.float32)  # an integer source dtype would quantise the smoothed values
        nib.save(nib.Nifti1Image(smoothed, img.affine, header), out_path)
        # one row per run as it finishes, so a crashed job still leaves a usable log
        record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "source_path": str(source_path), "output_path": str(out_path), "status": "ok"}
        pd.DataFrame([record]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
        log.info("[%d/%d] %s", i, len(runs), out_path.name)
    log.info("done, log -> %s", log_path)
