"""Unpaired motion-grade dataset: domain A = pooled Grades 2-6 (corrupted), domain B = Grade 1 (clean)."""
import csv
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

RunKey = tuple[str, str, str, str]  # (subject_id, session_id, run_id, task)


def run_key(row: dict) -> RunKey:
    """Key into the run-stats table for a chunk-metadata or run-stats CSV row."""
    return (row["subject_id"], row["session_id"], row["run_id"], row["task"])


def load_run_stats(run_stats_csv: str) -> dict[RunKey, tuple[float, float]]:
    """Per-run robust normalisation stats.

    Args:
        run_stats_csv: CSV with subject_id, session_id, run_id, task, median, scale.

    Returns:
        (subject_id, session_id, run_id, task) -> (median, scale).
    """
    with open(run_stats_csv) as f:
        return {run_key(row): (float(row["median"]), float(row["scale"])) for row in csv.DictReader(f)}


def load_split_assignment(splits_csv: str) -> tuple[tuple[str, ...], dict[tuple[str, ...], str]]:
    """Read a frozen split file.

    Args:
        splits_csv: Subject-level split (subject_id, split) or legacy split (subject_id, task, split).

    Returns:
        (key columns, key -> split); keys are (subject_id,) or (subject_id, task) to match the file.
    """
    with open(splits_csv) as f:
        rows = list(csv.DictReader(f))
    key_cols = tuple(c for c in ("subject_id", "task") if c in rows[0])
    return key_cols, {tuple(row[c] for c in key_cols): row["split"] for row in rows}


def load_chunk_rows(chunk_metadata_csv: str, splits_csv: str, split: str,
                    task: str | None = None) -> dict[str, list[dict]]:
    """Chunk-metadata rows that belong to one split.

    Args:
        chunk_metadata_csv: One row per chunk (grade, task, subject_id, chunk_path, ...).
        splits_csv: Frozen split file; the split column of chunk_metadata_csv itself is ignored.
        split: "train", "val" or "test".
        task: Keep only this task ("videos", "rest10"); None keeps all.

    Returns:
        grade -> rows, in file order.
    """
    key_cols, assignment = load_split_assignment(splits_csv)
    rows_by_grade: dict[str, list[dict]] = {}
    with open(chunk_metadata_csv) as f:
        for row in csv.DictReader(f):
            if assignment[tuple(row[c] for c in key_cols)] != split:
                continue
            if task is not None and row["task"] != task:
                continue
            rows_by_grade.setdefault(row["grade"], []).append(row)
    return rows_by_grade


def pad_axis0(x: Tensor, target: int) -> Tensor:
    """Zero-pad axis H symmetrically so three stride-2 downsamplings divide evenly (60 -> 64).

    Args:
        x: (..., T, H, W, D).
        target: Padded size of H.

    Returns:
        (..., T, target, W, D).
    """
    total = target - x.shape[-3]
    return F.pad(x, (0, 0, 0, 0, total // 2, total - total // 2))


def crop_axis0(x: Tensor, size: int) -> Tensor:
    """Inverse of pad_axis0: centre-crop H back to `size`.

    Args:
        x: (..., T, H, W, D) with H >= size.
        size: Original size of H.

    Returns:
        (..., T, size, W, D).
    """
    lo = (x.shape[-3] - size) // 2
    return x[..., lo:lo + size, :, :]


def normalize_volume(data: np.ndarray, median: float, scale: float) -> np.ndarray:
    """Robust per-run normalisation (x - median) / scale, inside the brain only.

    Args:
        data: (X, Y, Z, T) raw BOLD; brain = voxels nonzero at t=0.
        median: Run median.
        scale: Run scale.

    Returns:
        (X, Y, Z, T) float32, background exactly 0.
    """
    mask = data[..., 0] != 0
    out = np.zeros_like(data, dtype=np.float32)
    out[mask] = (data[mask] - median) / scale
    return out


def denormalize_chunk(chunk: Tensor, median: Tensor, scale: Tensor, mask: Tensor) -> Tensor:
    """Undo normalize_volume per sample, inside the brain only.

    Args:
        chunk: (B, T, H, W, D) normalised chunk (input or model output).
        median: (B,) run medians.
        scale: (B,) run scales.
        mask: (B, T, H, W, D) bool brain mask. Must come from the input (x_a != 0), not the model output,
            whose background is not guaranteed to be 0.

    Returns:
        (B, T, H, W, D) in BOLD units; voxels outside `mask` are 0, not `median`.
    """
    shape = (chunk.shape[0], 1, 1, 1, 1)
    median = median.to(chunk.dtype).view(shape).expand_as(chunk)
    scale = scale.to(chunk.dtype).view(shape).expand_as(chunk)
    out = torch.zeros_like(chunk)
    out[mask] = chunk[mask] * scale[mask] + median[mask]
    return out


class FMRIUnpairedGradeDataset(Dataset):
    """Unpaired A (pooled grades_b) / B (clean_grade) chunks of one split.

    Epoch length is min(A, B): the smaller domain is used in full (DataLoader shuffles it) and the
    larger one gets a fresh permutation per epoch via set_epoch(), which must be called before the
    epoch's iterator is created. full_coverage (val/test) cycles index % size through both domains.

    Args:
        split: "train", "val" or "test".
        chunk_metadata_csv: One row per chunk.
        run_stats_csv: Per-run (median, scale) for normalisation.
        splits_csv: Frozen split file.
        clean_grade: Grade used as domain B, e.g. "Grade 1".
        grades_b: Grades pooled into domain A (corrupted), despite the name.
        padded_h: Axis-0 size after zero-padding.
        task: Keep only this task; None keeps all.
        flip_prob: Probability of a left-right flip per chunk (train augmentation).
        base_seed: Seed of the per-epoch permutation, offset by the epoch.
        full_coverage: Iterate max(A, B) items so every chunk of both domains is seen (val/test).

    Raises:
        ValueError: If a domain is empty or a run has no normalisation stats.

    Items are dicts: "A", "B": (T, padded_h, W, D); "A_paths", "B_paths"; "A_meta", "B_meta" (see _meta).
    """

    def __init__(self, split: str, chunk_metadata_csv: str, run_stats_csv: str, splits_csv: str,
                 clean_grade: str, grades_b: list[str], padded_h: int, task: str | None = None,
                 flip_prob: float = 0.0, base_seed: int = 0, full_coverage: bool = False):
        super().__init__()
        self.flip_prob = flip_prob
        self.base_seed = base_seed
        self.full_coverage = full_coverage
        self.padded_h = padded_h
        self.run_stats = load_run_stats(run_stats_csv)

        rows_by_grade = load_chunk_rows(chunk_metadata_csv, splits_csv, split, task)
        self.A_rows = [row for grade in grades_b for row in rows_by_grade.get(grade, [])]
        self.B_rows = rows_by_grade.get(clean_grade, [])
        self.A_size, self.B_size = len(self.A_rows), len(self.B_rows)
        if self.A_size == 0 or self.B_size == 0:
            raise ValueError(f"split={split!r}: A_size={self.A_size}, B_size={self.B_size}, both must be > 0")
        missing = {run_key(r) for r in self.A_rows + self.B_rows} - self.run_stats.keys()
        if missing:
            raise ValueError(f"{len(missing)} runs have no normalisation stats, e.g. {next(iter(missing))}")

        self._b_is_larger = self.B_size >= self.A_size
        self._epoch_len = max(self.A_size, self.B_size) if full_coverage else min(self.A_size, self.B_size)
        self._larger_size = self.B_size if self._b_is_larger else self.A_size
        self.set_epoch(0)

    def __len__(self) -> int:
        return self._epoch_len

    def set_epoch(self, epoch: int) -> None:
        """Draw the larger domain's subset for this epoch; seeded, so identical on every DDP rank."""
        if self.full_coverage:
            return
        g = torch.Generator().manual_seed(self.base_seed + epoch)
        self._larger_epoch_indices = torch.randperm(self._larger_size, generator=g)[: self._epoch_len].tolist()

    def _load(self, row: dict) -> tuple[Tensor, float, float]:
        """Chunk -> ((T, padded_h, W, D) normalised tensor, run median, run scale)."""
        median, scale = self.run_stats[run_key(row)]
        data = np.asarray(nib.load(row["chunk_path"]).dataobj, dtype=np.float32)  # (H, W, D, T)
        x = torch.from_numpy(normalize_volume(data, median, scale)).permute(3, 0, 1, 2)
        x = pad_axis0(x, self.padded_h)
        if self.flip_prob > 0.0 and random.random() < self.flip_prob:
            x = torch.flip(x, dims=[1])  # left-right
        return x, median, scale

    @staticmethod
    def _meta(row: dict, median: float, scale: float) -> dict:
        """Per-chunk metadata kept with the batch; median/scale are needed to denormalise for metrics."""
        return dict(
            subject_id=row["subject_id"], age_group=row["age_group"], session_id=row["session_id"],
            run_id=row["run_id"], task=row["task"], grade=row["grade"],
            chunk_start=int(row["chunk_start"]), chunk_end=int(row["chunk_end"]),
            chunk_mean_fd=float(row["chunk_mean_fd"]), chunk_max_fd=float(row["chunk_max_fd"]),
            source_volume_path=row["source_volume_path"], median=median, scale=scale,
        )

    def __getitem__(self, index: int) -> dict:
        if self.full_coverage:
            a_row, b_row = self.A_rows[index % self.A_size], self.B_rows[index % self.B_size]
        elif self._b_is_larger:
            a_row, b_row = self.A_rows[index], self.B_rows[self._larger_epoch_indices[index]]
        else:
            a_row, b_row = self.A_rows[self._larger_epoch_indices[index]], self.B_rows[index]
        a, a_median, a_scale = self._load(a_row)
        b, b_median, b_scale = self._load(b_row)
        return {
            "A": a, "B": b,
            "A_paths": a_row["chunk_path"], "B_paths": b_row["chunk_path"],
            "A_meta": self._meta(a_row, a_median, a_scale), "B_meta": self._meta(b_row, b_median, b_scale),
        }
