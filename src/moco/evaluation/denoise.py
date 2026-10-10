"""Whole-run motion correction: normalise a run, correct it chunk by chunk, write it back in BOLD units."""
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from moco.data.grade import crop_axis0, load_run_stats, normalize_volume, pad_axis0
from moco.models.build import config_from_checkpoint, load_model
from moco.utils.run_info import git_state

log = logging.getLogger(__name__)


@torch.no_grad()
def correct_run(volume: np.ndarray, median: float, scale: float, model: torch.nn.Module, device: str,
                chunk_t: int, padded_h: int) -> np.ndarray:
    """Correct one full run with the A->B generator, chunk by chunk.

    Args:
        volume: (X, Y, Z, T) raw BOLD, brain-masked (background exactly 0).
        median: Run median used for the robust normalisation.
        scale: Run scale used for the robust normalisation.
        model: Trained CycleGAN in eval mode; must expose `correct(x)` with x: (1, chunk_t, H, Y, Z).
        device: Device the model lives on.
        chunk_t: Volumes per chunk, as trained (read from the checkpoint).
        padded_h: Size axis 0 is zero-padded to before the model, then cropped back.

    Returns:
        (X, Y, Z, T) corrected BOLD, same shape as `volume`; background stays exactly 0.
    """
    X, T = volume.shape[0], volume.shape[-1]
    mask = volume[..., 0] != 0  # brain = nonzero at t=0, same rule as training
    vol = torch.from_numpy(normalize_volume(volume, median, scale)).permute(3, 0, 1, 2)  # (T, X, Y, Z)

    corrected = []
    for start in range(0, T, chunk_t):  # non-overlapping chunks: 0..4, 5..9, ...
        chunk = vol[start:start + chunk_t]
        n_real = chunk.shape[0]
        if n_real < chunk_t:
            # model needs exactly chunk_t volumes: repeat the last one, drop the filler after correction
            chunk = torch.cat([chunk, chunk[-1:].repeat(chunk_t - n_real, 1, 1, 1)])
        out = model.correct(pad_axis0(chunk.unsqueeze(0).to(device), padded_h))
        corrected.append(crop_axis0(out, X)[0, :n_real].cpu().numpy())

    corrected_norm = np.concatenate(corrected)  # (T, X, Y, Z)
    # denormalise inside the input brain only; the model's background is not guaranteed to be 0
    bold = np.zeros_like(corrected_norm)
    mask_t = np.broadcast_to(mask, corrected_norm.shape)
    bold[mask_t] = corrected_norm[mask_t] * scale + median
    return bold.transpose(1, 2, 3, 0)


def corrected_path(source_volume_path: Path, source_root: Path, output_root: Path) -> Path:
    """Where a denoised run lives: the source's path relative to source_root, under output_root,
    with a `_corrected` suffix (the convention of both this repo and the original one)."""
    relative = source_volume_path.parent.relative_to(source_root)
    return output_root / relative / source_volume_path.name.replace(".nii.gz", "_corrected.nii.gz")


def select_runs(chunk_metadata_csv: str, which: str) -> pd.DataFrame:
    """Pick the video runs to denoise from the chunk metadata.

    Args:
        chunk_metadata_csv: Chunk metadata CSV; one row per chunk, so runs are deduplicated here.
        which: "first_2mo" = first session/run of each 2-month subject (9-month visits end in "A"),
            "all_2mo" = every video run of the 2-month subjects (what ISC needs),
            "all_video" = every video run at both ages.

    Returns:
        One row per run with subject_id, session_id, run_id, source_volume_path, fd_path, tr_seconds,
        sorted by subject.

    Raises:
        ValueError: If `which` is not one of the three options.
    """
    meta = pd.read_csv(chunk_metadata_csv, dtype={"session_id": str, "run_id": str})  # keep "002", not 2
    video = meta[meta["task"] == "videos"]
    cols = ["subject_id", "session_id", "run_id", "source_volume_path", "fd_path", "tr_seconds"]
    if which == "all_video":
        return video[cols].drop_duplicates().sort_values(["subject_id", "session_id", "run_id"])
    video = video[(video["age_group"] == "2mo") & ~video["subject_id"].str.endswith("A")]
    runs = video[cols].drop_duplicates().sort_values(["subject_id", "session_id", "run_id"])
    if which == "all_2mo":
        return runs
    if which == "first_2mo":
        return runs.groupby("subject_id", as_index=False).first()
    raise ValueError(f"unknown run selection {which!r}")


def check_output_root(output_root: Path, overwrite: bool) -> None:
    """Stop before anything is written if the output folder already holds files.

    Args:
        output_root: Folder the denoised runs will be written to.
        overwrite: If True, allow writing into a non-empty folder.

    Raises:
        SystemExit: If `output_root` is non-empty and `overwrite` is False.
    """
    # Hydra creates the (empty) run dir before main(), so test for contents, not existence
    if overwrite or not output_root.exists() or not any(output_root.iterdir()):
        return
    raise SystemExit(f"{output_root} already exists and is not empty. Choose a new output_name=<name>, "
                     f"or pass overwrite=true to overwrite its runs.")


def write_manifest(output_root: Path, checkpoint: Path, epoch: int, chunk_t: int, padded_h: int,
                   runs: pd.DataFrame, cfg: DictConfig) -> None:
    """Describe how the denoised folder was made in output_root/manifest.json.

    Args:
        output_root: Denoise output folder.
        checkpoint: Checkpoint used; its sha256 pins the exact weights.
        epoch: Checkpoint epoch.
        chunk_t: Volumes per chunk, as read from the checkpoint.
        padded_h: Padded axis-0 size, as read from the checkpoint.
        runs: Selected runs, one row each with subject_id, session_id, run_id.
        cfg: Composed `configs/denoise.yaml`.
    """
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_epoch": epoch,
        "chunk_timepoints": chunk_t,
        "padded_axis0": padded_h,
        "runs": cfg.runs,
        "n_runs": len(runs),
        "source_root": str(cfg.source_root),
        "run_stats_csv": str(cfg.data.run_stats_csv),
        "chunk_metadata_csv": str(cfg.data.chunk_metadata_csv),
        **git_state(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run_denoise(cfg: DictConfig) -> None:
    """Denoise every selected run and save it under output_root, mirroring the source directory tree.

    Args:
        cfg: Composed `configs/denoise.yaml`; uses checkpoint, device, runs, source_root, output_root and
            data.run_stats_csv / data.chunk_metadata_csv.
    """
    model, ckpt = load_model(cfg.checkpoint, cfg.device)
    # chunk length and padded size come from the checkpoint, so old-repo checkpoints work unchanged
    ckpt_cfg = config_from_checkpoint(ckpt)
    chunk_t, padded_h = ckpt_cfg.data.in_timepoints, ckpt_cfg.data.spatial_dims[0]
    run_stats = load_run_stats(cfg.data.run_stats_csv)
    runs = select_runs(cfg.data.chunk_metadata_csv, cfg.runs)
    source_root, output_root = Path(cfg.source_root), Path(cfg.output_root)
    log.info("denoising %d runs with %s (epoch %d)", len(runs), cfg.checkpoint, ckpt["epoch"])
    write_manifest(output_root, Path(cfg.checkpoint), ckpt["epoch"], chunk_t, padded_h, runs, cfg)
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
        out_path = corrected_path(Path(row.source_volume_path), source_root, output_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(corrected, img.affine, img.header), out_path)  # keep source affine/header
        append_log({**record, "output_path": str(out_path), "status": "ok"})
        log.info("[%d/%d] %s -> %s", i, len(runs), row.subject_id, out_path)
    log.info("done, log -> %s", log_path)
