"""Old vs new pipeline QC (raw, st_v4): python runs/_equivalence/pipeline_qc/compare.py > <this dir>/result.txt"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OLD = Path("/lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion-artefacts-correction/"
           "motion_denoising_pipeline_level_evaluation")
NEW = Path(__file__).resolve().parents[3] / "motion_denoising_qc"
# (old QC folder, old FC-significance folder, new folder)
PAIRS = [("raw_data_each_sub_first_run", "raw_fc_edges_significance", "original_data"),
         ("st_v4_corrected_each_sub_first_run", "corrected_fc_edges_significance", "motion_corrected_st_v4")]

for old_qc, old_fc, new_name in PAIRS:
    new = NEW / new_name
    print(f"===== {old_qc} vs {new_name}")
    old_subj, new_subj = pd.read_csv(OLD / old_qc / "manifest.csv"), pd.read_csv(new / "per_subject.csv")
    print("same subjects in same order:", old_subj["subject_id"].tolist() == new_subj["subject_id"].tolist())
    for col in ("mean_fd", "Q", "dvars", "tsnr"):
        print(f"{col:8s} max |old - new| = {np.abs(old_subj[col].to_numpy() - new_subj[col].to_numpy()).max():.3e}")
    for name, old_dir in (("qc_fc_r", old_qc), ("qc_fc_p", old_qc), ("distance_matrix", old_qc),
                          ("fc_mean_z", old_fc), ("fc_p", old_fc)):
        diff = np.nanmax(np.abs(np.load(OLD / old_dir / f"{name}.npy") - np.load(new / f"{name}.npy")))
        print(f"{name:16s} max |old - new| = {diff:.3e}")
    for name, old_dir in (("qc_fc_fdr_significant", old_qc), ("fc_significant", old_fc)):
        old, cur = np.load(OLD / old_dir / f"{name}.npy"), np.load(new / f"{name}.npy")
        print(f"{name:22s} old {old.sum() // 2} new {cur.sum() // 2} edges, {(old != cur).sum() // 2} differ")
    print("\nold summary.txt:\n" + (OLD / old_qc / "summary.txt").read_text())
    print("old fc_significance_summary.txt:\n" + (OLD / old_fc / "fc_significance_summary.txt").read_text())
    print("new summary.json:\n" + json.dumps(json.loads((new / "summary.json").read_text()), indent=2) + "\n")
