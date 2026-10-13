"""Old vs new voxel-wise ISC (st_v4, order A): python runs/_equivalence/isc_voxel_wise/compare.py > result.txt"""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

OLD = Path("/lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion-artefacts-correction/"
           "motion_denoising_pipeline_level_evaluation/voxel_wise_isc_loso/st_v4/order_A")
NEW = (Path(__file__).resolve().parents[3] / "motion_denoising_qc" / "motion_corrected_st_v4_all_video_runs"
       / "isc_voxel_wise" / "order_A")


def max_abs_diff(name: str) -> float:
    old, new = nib.load(OLD / name).get_fdata(), nib.load(NEW / name).get_fdata()
    assert old.shape == new.shape, f"{name}: shape {old.shape} vs {new.shape}"
    assert np.array_equal(np.isnan(old), np.isnan(new)), f"{name}: NaN pattern differs"
    return float(np.nanmax(np.abs(old - new)))


masks_equal = np.array_equal(nib.load(OLD / "common_mask.nii.gz").get_fdata(),
                             nib.load(NEW / "common_mask.nii.gz").get_fdata())
print(f"common mask identical: {masks_equal}")
subjects = [pd.read_csv(root / "subject_maps" / "subjects.csv") for root in (OLD, NEW)]
print(f"subjects identical: {subjects[0].equals(subjects[1])}")
for name in ("raw", "corrected"):
    print(f"subject_isc_z_{name:10s} max |old - new| = {max_abs_diff(f'subject_maps/subject_isc_z_{name}.nii.gz'):.3e}")
for name in ("raw", "corrected", "corrected_minus_raw"):
    for key in ("mean_z", "ci_low_z", "ci_high_z", "q_fdr", "mean_z_fdr"):
        print(f"{name}_{key:12s} max |old - new| = {max_abs_diff(f'group_maps/{name}_{key}.nii.gz'):.3e}")
for table in ("group_summary.csv", "network_isc.csv"):
    old, new = pd.read_csv(OLD / "tables" / table), pd.read_csv(NEW / "tables" / table)
    numeric = old.select_dtypes("number").columns
    same_keys = old.drop(columns=numeric).equals(new.drop(columns=numeric))
    print(f"{table}: same rows {same_keys}, max |old - new| = "
          f"{np.abs(old[numeric].to_numpy() - new[numeric].to_numpy()).max():.3e}")
print("\nold group_summary.csv:\n" + (OLD / "tables" / "group_summary.csv").read_text())
print("new group_summary.csv:\n" + (NEW / "tables" / "group_summary.csv").read_text())
