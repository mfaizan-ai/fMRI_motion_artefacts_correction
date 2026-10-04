"""Freeze a subject-level split: python scripts/make_splits.py [name=...] [chunk_metadata_csv=...]"""
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from moco.data.splits import assign_splits, subject_profiles
from moco.utils.logs import setup_logging
from moco.utils.run_info import git_state

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="splits", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    out_csv = Path(cfg.output_csv)
    if out_csv.exists() and not cfg.overwrite:
        raise FileExistsError(f"{out_csv} exists; splits are frozen (pass overwrite=true to replace)")

    meta = pd.read_csv(cfg.chunk_metadata_csv, dtype=str)
    profiles = subject_profiles(meta, list(cfg.stratify.high_motion_grades), cfg.stratify.n_motion_bins)
    split = assign_splits(profiles, dict(cfg.ratios), cfg.seed)
    split.reset_index().to_csv(out_csv, index=False)

    meta["split"] = meta["subject_id"].map(split)
    profiles["split"] = profiles["subject_id"].map(split)
    manifest = {
        "source": cfg.chunk_metadata_csv,
        "source_sha256": hashlib.sha256(Path(cfg.chunk_metadata_csv).read_bytes()).hexdigest(),
        "unit": "subject_id",
        "config": OmegaConf.to_container(cfg, resolve=True),
        "subjects": profiles["split"].value_counts().to_dict(),
        "subjects_by_age": profiles.groupby(["split", "age_group"]).size().unstack(fill_value=0).to_dict("index"),
        "chunks_by_grade": meta.groupby(["split", "grade"]).size().unstack(fill_value=0).to_dict("index"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **git_state(),
    }
    out_csv.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("wrote %s: %s", out_csv, manifest["subjects"])


if __name__ == "__main__":
    main()
