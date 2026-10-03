"""Freeze the (subject_id, task) -> split assignment from a chunk metadata CSV into splits/, once.

python scripts/freeze_splits.py <chunk_metadata.csv> splits/<name>.csv
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from moco.utils.run_info import git_state


def main(chunk_metadata_csv: str, out_csv: str) -> None:
    meta = pd.read_csv(chunk_metadata_csv, dtype=str)
    assignment = meta[["subject_id", "task", "split"]].drop_duplicates()
    if assignment.duplicated(["subject_id", "task"]).any():
        raise ValueError("a (subject_id, task) pair spans several splits")
    out = Path(out_csv)
    assignment.sort_values(["subject_id", "task"]).to_csv(out, index=False)
    manifest = {
        "source": chunk_metadata_csv,
        "source_sha256": hashlib.sha256(Path(chunk_metadata_csv).read_bytes()).hexdigest(),
        "granularity": "subject_id x task",
        "subjects_in_several_splits": int((assignment.groupby("subject_id")["split"].nunique() > 1).sum()),
        "rows_per_split": assignment["split"].value_counts().to_dict(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **git_state(),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
