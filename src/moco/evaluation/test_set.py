"""Correct the held-out test chunks and summarise the validation fMRI metrics by motion grade."""
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from moco.data.grade import denormalize_chunk, load_chunk_rows, load_run_stats, normalize_volume, pad_axis0, run_key
from moco.evaluation.metrics import fmri_metrics
from moco.models.build import load_model

log = logging.getLogger(__name__)


def run_test(cfg) -> None:
    """Test split, video task, 2-month subjects only (as in the original evaluation)."""
    model, ckpt = load_model(cfg.checkpoint, cfg.device)
    run_stats = load_run_stats(cfg.data.run_stats_csv)
    rows_by_grade = load_chunk_rows(cfg.data.chunk_metadata_csv, cfg.data.splits_csv, "test", task="videos")
    log.info("testing %s (epoch %d)", cfg.checkpoint, ckpt["epoch"])

    records = []
    for grade, rows in rows_by_grade.items():
        rows = [r for r in rows if r["age_group"] == "2mo" and not r["subject_id"].endswith("A")]
        for row in rows:
            median, scale = run_stats[run_key(row)]
            data = np.asarray(nib.load(row["chunk_path"]).dataobj, dtype=np.float32)
            x_in = torch.from_numpy(normalize_volume(data, median, scale)).permute(3, 0, 1, 2)
            x_in = pad_axis0(x_in, cfg.data.spatial_dims[0]).unsqueeze(0).to(cfg.device)
            with torch.no_grad():
                x_out = model.correct(x_in)
            median_t = torch.tensor([median], device=cfg.device)
            scale_t = torch.tensor([scale], device=cfg.device)
            brain = x_in != 0
            metrics = fmri_metrics(denormalize_chunk(x_in, median_t, scale_t, brain),
                                   denormalize_chunk(x_out, median_t, scale_t, brain))
            records.append({**metrics, "grade": grade, "subject_id": row["subject_id"],
                            "chunk_path": row["chunk_path"]})
        log.info("%s: %d chunks", grade, len(rows))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "test_metrics_per_chunk.csv", index=False)
    summary = df.groupby("grade")[["dvars_improvement", "tsnr_improvement", "gs_std_improvement",
                                   "smoothness_ratio"]].agg(["mean", "std"])
    summary.to_csv(out_dir / "test_metrics_summary_by_grade.csv")
    log.info("\n%s", summary)
