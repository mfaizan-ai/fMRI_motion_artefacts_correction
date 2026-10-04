"""Whole-run motion correction: normalise a run, correct it chunk by chunk, write it back in BOLD units."""
import logging
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from moco.data.grade import crop_axis0, load_run_stats, normalize_volume, pad_axis0
from moco.models.build import config_from_checkpoint, load_model
from moco.utils.run_info import git_state

log = logging.getLogger(__name__)


@torch.no_grad()
def correct_run(volume: np.ndarray, median: float, scale: float, model: torch.nn.Module, device: str,
                chunk_t: int, padded_h: int) -> np.ndarray:
    """volume: (X, Y, Z, T) raw BOLD -> corrected (X, Y, Z, T) BOLD.

    Non-overlapping chunk_t-volume chunks; a short final chunk is filled by repeating its last volume
    and the filler is dropped after correction. Background stays exactly 0."""
    X, T = volume.shape[0], volume.shape[-1]
    mask = volume[..., 0] != 0
    vol = torch.from_numpy(normalize_volume(volume, median, scale)).permute(3, 0, 1, 2)  # (T, X, Y, Z)

    corrected = []
    for start in range(0, T, chunk_t):
        chunk = vol[start:start + chunk_t]
        n_real = chunk.shape[0]
        if n_real < chunk_t:
            chunk = torch.cat([chunk, chunk[-1:].repeat(chunk_t - n_real, 1, 1, 1)])
        out = model.correct(pad_axis0(chunk.unsqueeze(0).to(device), padded_h))
        corrected.append(crop_axis0(out, X)[0, :n_real].cpu().numpy())

    corrected_norm = np.concatenate(corrected)  # (T, X, Y, Z)
    bold = np.zeros_like(corrected_norm)
    mask_t = np.broadcast_to(mask, corrected_norm.shape)
    bold[mask_t] = corrected_norm[mask_t] * scale + median
    return bold.transpose(1, 2, 3, 0)


def select_runs(chunk_metadata_csv: str, which: str) -> pd.DataFrame:
    """Video runs to denoise: "first_2mo" = first session/run of each 2-month subject, "all_video" = every run."""
    meta = pd.read_csv(chunk_metadata_csv, dtype={"session_id": str, "run_id": str})
    video = meta[meta["task"] == "videos"]
    cols = ["subject_id", "session_id", "run_id", "source_volume_path"]
    if which == "all_video":
        return video[cols].drop_duplicates().sort_values(["subject_id", "session_id", "run_id"])
    if which == "first_2mo":
        video = video[(video["age_group"] == "2mo") & ~video["subject_id"].str.endswith("A")]
        runs = video[cols].drop_duplicates().sort_values(["subject_id", "session_id", "run_id"])
        return runs.groupby("subject_id", as_index=False).first()
    raise ValueError(f"unknown run selection {which!r}")


def check_output_root(output_root: Path, overwrite: bool) -> None:
    """Exit if output_root already holds files, unless overwrite is set."""
    # Hydra creates the (empty) run dir before main(), so test for contents, not existence
    if overwrite or not output_root.exists() or not any(output_root.iterdir()):
        return
    raise SystemExit(f"{output_root} already exists and is not empty. Choose a new output_name=<name>, "
                     f"or pass overwrite=true to overwrite its runs.")


def run_denoise(cfg) -> None:
    model, ckpt = load_model(cfg.checkpoint, cfg.device)
    ckpt_cfg = config_from_checkpoint(ckpt)
    chunk_t, padded_h = ckpt_cfg.data.in_timepoints, ckpt_cfg.data.spatial_dims[0]
    run_stats = load_run_stats(cfg.data.run_stats_csv)
    runs = select_runs(cfg.data.chunk_metadata_csv, cfg.runs)
    source_root, output_root = Path(cfg.source_root), Path(cfg.output_root)
    log.info("denoising %d runs with %s (epoch %d)", len(runs), cfg.checkpoint, ckpt["epoch"])
    log_path = output_root / "denoise_log.csv"
    provenance = {"checkpoint": str(cfg.checkpoint), **git_state()}

    def append_log(record: dict) -> None:
        # one row per run as it finishes, so a crashed job still leaves a usable log
        output_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{**record, **provenance}]).to_csv(log_path, mode="a", header=not log_path.exists(),
                                                        index=False)

    for i, row in enumerate(runs.itertuples(index=False), 1):
        record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "subject_id": row.subject_id, "session_id": row.session_id, "run_id": row.run_id,
                  "source_volume_path": row.source_volume_path}
        key = (row.subject_id, str(row.session_id), str(row.run_id), "videos")
        if key not in run_stats:
            log.warning("[%d/%d] %s: no normalisation stats, skipped", i, len(runs), row.subject_id)
            append_log({**record, "status": "skipped_no_run_stats"})
            continue
        img = nib.load(row.source_volume_path)
        corrected = correct_run(np.asarray(img.dataobj, dtype=np.float32), *run_stats[key], model, cfg.device,
                                chunk_t, padded_h)
        source = Path(row.source_volume_path)
        out_path = output_root / source.parent.relative_to(source_root) / source.name.replace(".nii.gz",
                                                                                              "_corrected.nii.gz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(corrected, img.affine, img.header), out_path)
        append_log({**record, "output_path": str(out_path), "status": "ok"})
        log.info("[%d/%d] %s -> %s", i, len(runs), row.subject_id, out_path)
    log.info("done, log -> %s", log_path)
