import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from moco.evaluation.smoothing import run_smoothing, smooth_run

AFFINE_2MM = np.diag([2.0, 2.0, 2.0, 1.0])


def test_fwhm_is_in_mm_and_time_points_stay_separate():
    """An impulse at one time point: half maximum at FWHM/2 = 1 voxel of 2 mm, nothing leaks into other volumes."""
    volume = np.zeros((21, 21, 21, 3), dtype=np.float32)
    volume[10, 10, 10, 1] = 1.0
    out = smooth_run(volume, AFFINE_2MM, fwhm_mm=4.0, preserve_zero_background=False)
    assert out.shape == volume.shape and out.dtype == np.float32
    assert np.all(out[..., [0, 2]] == 0)
    assert out[11, 10, 10, 1] / out[10, 10, 10, 1] == pytest.approx(0.5, rel=1e-3)


def test_zero_background_is_kept_only_when_asked():
    rng = np.random.default_rng(0)
    volume = np.zeros((12, 12, 12, 4), dtype=np.float32)
    volume[3:9, 3:9, 3:9] = rng.normal(size=(6, 6, 6, 4))
    kept = smooth_run(volume, AFFINE_2MM, fwhm_mm=4.0, preserve_zero_background=True)
    spread = smooth_run(volume, AFFINE_2MM, fwhm_mm=4.0, preserve_zero_background=False)
    background = ~np.any(volume != 0, axis=3)
    assert np.all(kept[background] == 0)
    assert np.any(spread[background] != 0)


def test_non_finite_run_is_rejected():
    volume = np.ones((4, 4, 4, 2), dtype=np.float32)
    volume[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        smooth_run(volume, AFFINE_2MM, fwhm_mm=4.0, preserve_zero_background=True)


def write_source_tree(source_root: Path, names: dict[str, str], units: str) -> None:
    for task, name in names.items():
        run_dir = source_root / "_subject_id_X" / f"_run_001_session_1_task_name_{task}"
        run_dir.mkdir(parents=True)
        img = nib.Nifti1Image(np.ones((6, 6, 6, 3), dtype=np.int16), AFFINE_2MM)
        img.header.set_xyzt_units(units, "sec")
        nib.save(img, run_dir / name)


def smoothing_cfg(tmp_path: Path) -> DictConfig:
    return OmegaConf.create({"source": "raw", "source_root": str(tmp_path / "source"), "denoised_root": str(tmp_path),
                             "output_root": str(tmp_path / "out"), "pattern": "**/*task-videos*.nii.gz",
                             "fwhm_mm": 4, "preserve_zero_background": True, "max_runs": None})


@pytest.mark.parametrize("units", ["mm", "unknown"])  # most real FLIRT outputs leave the units code unset
def test_run_smoothing_mirrors_tree_and_keeps_filenames(tmp_path, units):
    """Same relative path and filename as the input, so isc_voxel_wise.py finds runs through root overrides."""
    names = {"videos": "sub-X_ses-1_task-videos_run-001_bold.nii.gz",
             "rest10": "sub-X_ses-1_task-rest10_run-001_bold.nii.gz"}
    write_source_tree(tmp_path / "source", names, units)
    run_smoothing(smoothing_cfg(tmp_path))

    outputs = sorted(p.relative_to(tmp_path / "out") for p in (tmp_path / "out").rglob("*.nii.gz"))
    assert [str(p) for p in outputs] == [f"_subject_id_X/_run_001_session_1_task_name_videos/{names['videos']}"]
    assert nib.load(tmp_path / "out" / outputs[0]).get_data_dtype() == np.float32
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert (manifest["fwhm_mm"], manifest["n_runs"], manifest["source"]) == (4.0, 1, "raw")
    assert pd.read_csv(tmp_path / "out" / "smooth_log.csv")["status"].tolist() == ["ok"]


def test_run_smoothing_rejects_non_mm_units(tmp_path):
    write_source_tree(tmp_path / "source", {"videos": "sub-X_task-videos_bold.nii.gz"}, "meter")
    with pytest.raises(ValueError, match="not mm"):
        run_smoothing(smoothing_cfg(tmp_path))


def test_output_name_records_source_and_fwhm():
    from hydra import compose, initialize

    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose("smooth", overrides=["source=raw", "fwhm_mm=6"])
    assert cfg.output_root.endswith("/smoothed_data/raw_fwhm-6mm")
