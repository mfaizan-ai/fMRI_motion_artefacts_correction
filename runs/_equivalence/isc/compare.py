"""Old vs new leave-one-out ISC (raw, st_v4; orders A-F): python runs/_equivalence/isc/compare.py > result.txt"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

OLD_REPO = Path("/lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion-artefacts-correction")
OLD_DATA = Path("/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/isc_segmenting/"
                "isc_comparison_data_cyclegans")
NEW = Path(__file__).resolve().parents[3] / "motion_denoising_qc"
# (old time-course tree, new source folder): the old trees cut chunks from the same cropped runs
PAIRS = [("train_data_raw_preprocessed", "original_data"), ("denoised_st_v4", "motion_corrected_st_v4_all_video_runs")]

# the old repo's own ISC functions, imported read-only
spec = importlib.util.spec_from_file_location("old_loo_isc", OLD_REPO / "ISC_analysis" / "loo_isc.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

for old_tree, new_source in PAIRS:
    for order in "ABCDEF":
        tc_root = OLD_DATA / old_tree / order / "time_courses"
        rows = list(pd.read_csv(tc_root / f"order_{order}_timecourses_manifest.csv", dtype=str).to_dict("records"))
        series = old.load_all_series(rows)
        new_dir = NEW / new_source / "isc_network" / f"order_{order}"
        print(f"===== {old_tree} vs {new_source}, order {order}")
        for key in ("roi", "network"):
            subjects, old_isc = old.compute_all_isc(series, key)
            new = pd.read_csv(new_dir / f"{key}_isc_per_subject.csv", index_col="subject")
            same = list(new.index) == subjects
            diff = np.nanmax(np.abs(new.to_numpy() - old_isc)) if same else float("nan")
            print(f"{key:8s} subjects {len(subjects)} same order {same}  per-subject ISC max |old - new| = {diff:.3e}")
        _, old_isfc = old.compute_all_isfc(series, "network")
        old_group = old.symmetrize(old.group_isfc_summary(old_isfc))
        new_group = pd.read_csv(new_dir / "network_isfc_group.csv", index_col=0).to_numpy()
        print(f"network ISFC group max |old - new| = {np.abs(new_group - old_group).max():.3e}")
