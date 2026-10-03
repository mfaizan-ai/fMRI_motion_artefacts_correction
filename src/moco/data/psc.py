"""Legacy PSC-normalised datasets over cyclegans_chunk5_dataset (A_corrupted / B_motion_free folders),
plus the sequence datasets used for temporal-consistency and FC losses.

Volumes are trilinearly resampled to `target_spatial` and normalised to percent signal change.
"""
import csv
import json
import logging
import os
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler

log = logging.getLogger(__name__)

PSC_EPS = 1e-6
BRAIN_FRACTION_OF_MAX = 0.01  # brain = temporal mean > 1% of the chunk's max mean (drops boundary interpolation)


def load_nifti(path: str, target_spatial: tuple) -> tuple[Tensor, tuple]:
    """(X, Y, Z, T) on disk -> (T, *target_spatial) tensor and the original (X, Y, Z)."""
    data = np.asarray(nib.load(path, mmap=True).dataobj, dtype=np.float32).transpose(3, 0, 1, 2)
    orig_shape = tuple(data.shape[1:])
    x = torch.from_numpy(data)
    if orig_shape != tuple(target_spatial):
        x = F.interpolate(x.unsqueeze(0), size=tuple(target_spatial), mode="trilinear",
                          align_corners=False).squeeze(0)
    return x, orig_shape


def psc_normalise(x: Tensor) -> tuple[Tensor, Tensor]:
    """x: (T, X, Y, Z) -> PSC (T, X, Y, Z) with background zeroed, and the temporal mean (1, X, Y, Z)."""
    mean_vol = x.mean(dim=0, keepdim=True)
    brain_mask = mean_vol > mean_vol.max() * BRAIN_FRACTION_OF_MAX
    psc = (x - mean_vol) / (mean_vol.abs() + PSC_EPS)
    return psc * brain_mask, mean_vol


def psc_denormalise(psc: Tensor, mean_vol: Tensor) -> Tensor:
    """Inverse of psc_normalise; mean_vol must broadcast against psc."""
    return psc * (mean_vol.abs() + PSC_EPS) + mean_vol


def _nifti_files(directory: str) -> list[str]:
    files = sorted(str(f) for f in Path(directory).iterdir() if ".nii" in f.name)
    if not files:
        raise FileNotFoundError(f"no NIfTI files in {directory}")
    return files


def _item(psc_a: Tensor, mean_a: Tensor, orig_a: tuple, path_a: str,
          psc_b: Tensor, mean_b: Tensor, orig_b: tuple, path_b: str) -> dict:
    return {"A": psc_a, "B": psc_b, "mean_vol_A": mean_a, "mean_vol_B": mean_b,
            "orig_shape_A": orig_a, "orig_shape_B": orig_b, "path_A": path_a, "path_B": path_b}


class TrainFMRIDataset(Dataset):
    """Unpaired training chunks. Epoch = len(B); A is consumed from a shuffled queue across epochs."""

    def __init__(self, root_dir: str, target_spatial: tuple, augment: bool = False):
        super().__init__()
        self.target_spatial = target_spatial
        self.augment = augment
        self.files_A = _nifti_files(os.path.join(root_dir, "A_corrupted"))
        self.files_B = _nifti_files(os.path.join(root_dir, "B_motion_free"))
        self._a_queue: list[int] = []
        self.on_epoch_start()
        log.info("TrainFMRIDataset: A=%d B=%d", len(self.files_A), len(self.files_B))

    def on_epoch_start(self) -> None:
        epoch_size = len(self.files_B)
        if len(self._a_queue) < epoch_size:
            self._a_queue = list(range(len(self.files_A)))
            random.shuffle(self._a_queue)
        self._a_epoch_indices = self._a_queue[:epoch_size]
        self._a_queue = self._a_queue[epoch_size:]
        self._b_order = list(range(len(self.files_B)))
        random.shuffle(self._b_order)

    def __len__(self) -> int:
        return len(self.files_B)

    def _load(self, path: str) -> tuple[Tensor, Tensor, tuple]:
        raw, orig_shape = load_nifti(path, self.target_spatial)
        return *psc_normalise(raw), orig_shape

    def __getitem__(self, idx: int) -> dict:
        path_a = self.files_A[self._a_epoch_indices[idx]]
        path_b = self.files_B[self._b_order[idx]]
        psc_a, mean_a, orig_a = self._load(path_a)
        psc_b, mean_b, orig_b = self._load(path_b)
        if self.augment:  # independent left-right flip per domain
            if random.random() > 0.5:
                psc_a, mean_a = torch.flip(psc_a, dims=[1]), torch.flip(mean_a, dims=[1])
            if random.random() > 0.5:
                psc_b, mean_b = torch.flip(psc_b, dims=[1]), torch.flip(mean_b, dims=[1])
        return _item(psc_a, mean_a, orig_a, path_a, psc_b, mean_b, orig_b, path_b)


class ValFMRIDataset(Dataset):
    """All of A against B shuffled once at construction (fixed pairing across epochs)."""

    def __init__(self, root_dir: str, target_spatial: tuple):
        super().__init__()
        self.target_spatial = target_spatial
        self.files_A = _nifti_files(os.path.join(root_dir, "A_corrupted"))
        self.files_B = _nifti_files(os.path.join(root_dir, "B_motion_free"))
        self._b_order = list(range(len(self.files_B)))
        random.shuffle(self._b_order)

    def on_epoch_start(self) -> None:
        pass

    def __len__(self) -> int:
        return min(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx: int) -> dict:
        path_a, path_b = self.files_A[idx], self.files_B[self._b_order[idx]]
        raw_a, orig_a = load_nifti(path_a, self.target_spatial)
        raw_b, orig_b = load_nifti(path_b, self.target_spatial)
        return _item(*psc_normalise(raw_a), orig_a, path_a, *psc_normalise(raw_b), orig_b, path_b)


class TestFMRIDataset(Dataset):
    """Domain A only, sorted, no augmentation."""

    def __init__(self, root_dir: str, target_spatial: tuple):
        self.target_spatial = target_spatial
        self.files_A = _nifti_files(os.path.join(root_dir, "A_corrupted"))

    def __len__(self) -> int:
        return len(self.files_A)

    def __getitem__(self, idx: int) -> dict:
        path = self.files_A[idx]
        raw, orig_shape = load_nifti(path, self.target_spatial)
        psc, mean_vol = psc_normalise(raw)
        return {"A": psc, "mean_vol_A": mean_vol, "orig_shape_A": orig_shape, "path_A": path}


def _chunk_path_lookup(chunk_metadata_csv: str) -> dict[tuple[str, int], str]:
    with open(chunk_metadata_csv) as f:
        return {(r["preprocessed_bold_file"], int(r["chunk_index"])): r["chunk_nifti_path"]
                for r in csv.DictReader(f)}


def _manifest_sequences(manifest_csv: str, split: str) -> list[dict]:
    with open(manifest_csv) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    if not rows:
        raise FileNotFoundError(f"no sequences for split {split!r} in {manifest_csv}")
    return rows


def load_normalised_sequence(chunk_paths: list[str], target_spatial: tuple) -> tuple[Tensor, Tensor, tuple]:
    """S ordered chunks with ONE PSC baseline over all S*T volumes -> (S, T, X, Y, Z), (S, 1, X, Y, Z).

    A per-chunk baseline would high-pass the signal at the chunk length and erase slow dynamics."""
    raws = [load_nifti(p, target_spatial) for p in chunk_paths]
    S, T = len(raws), raws[0][0].shape[0]
    psc, mean_vol = psc_normalise(torch.cat([r for r, _ in raws], dim=0))
    seq = psc.reshape(S, T, *psc.shape[1:])
    return seq, mean_vol.unsqueeze(0).expand(S, -1, -1, -1, -1).clone(), raws[0][1]


class SequenceDataset(Dataset):
    """Ordered video sequences (domain A) against random (train) or fixed (val) clean chunks (domain B).

    One item = one sequence of S chunks; use batch_size=1 so S acts as the model's batch."""

    def __init__(self, manifest_csv: str, chunk_metadata_csv: str, b_motion_free_dir: str, split: str,
                 target_spatial: tuple, train: bool, augment: bool = False):
        super().__init__()
        self.target_spatial = target_spatial
        self.train = train
        self.augment = augment
        self.sequences = _manifest_sequences(manifest_csv, split)
        self.path_lookup = _chunk_path_lookup(chunk_metadata_csv)
        self.files_B = _nifti_files(b_motion_free_dir)
        self._b_order = list(range(len(self.files_B)))
        if not train:
            random.shuffle(self._b_order)

    def on_epoch_start(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.sequences)

    def _chunk_paths(self, row: dict) -> list[str]:
        bold_file = row["preprocessed_bold_file"]
        return [self.path_lookup[(bold_file, c["chunk_index"])] for c in json.loads(row["ordered_chunk_paths"])]

    def __getitem__(self, idx: int) -> dict:
        row = self.sequences[idx]
        chunk_paths = self._chunk_paths(row)
        seq_a, mean_a, orig_a = load_normalised_sequence(chunk_paths, self.target_spatial)

        b_psc, b_mean = [], []
        for i in range(len(chunk_paths)):
            b_idx = (random.randrange(len(self.files_B)) if self.train
                     else self._b_order[(idx * len(chunk_paths) + i) % len(self.files_B)])
            raw, orig_b = load_nifti(self.files_B[b_idx], self.target_spatial)
            psc, mean = psc_normalise(raw)
            b_psc.append(psc)
            b_mean.append(mean)
        seq_b, mean_b = torch.stack(b_psc), torch.stack(b_mean)

        if self.augment:
            if random.random() > 0.5:
                seq_a, mean_a = torch.flip(seq_a, dims=[2]), torch.flip(mean_a, dims=[2])
            if random.random() > 0.5:
                seq_b, mean_b = torch.flip(seq_b, dims=[2]), torch.flip(mean_b, dims=[2])
        item = _item(seq_a, mean_a, orig_a, row["preprocessed_bold_file"], seq_b, mean_b, orig_b, "sequence")
        item["subject_id"] = row["subject_id"]
        return item


def build_psc_dataloaders(data_cfg, batch_size: int, num_workers: int, splits: list[str],
                          distributed: bool = False, world_size: int = 1, rank: int = 0) -> dict[str, DataLoader]:
    """Loaders for the legacy PSC data (flat chunks, or sequences when data_cfg.sequence_mode)."""
    target = tuple(data_cfg.spatial_dims)
    root = data_cfg.root
    if data_cfg.sequence_mode:
        batch_size = 1
        builders = {
            "train": lambda: SequenceDataset(data_cfg.manifest_csv, data_cfg.chunk_metadata_csv,
                                             os.path.join(root, "train", "B_motion_free"), "train", target,
                                             train=True, augment=data_cfg.augment),
            "val": lambda: SequenceDataset(data_cfg.manifest_csv, data_cfg.chunk_metadata_csv,
                                           os.path.join(root, "val", "B_motion_free"), "val", target, train=False),
        }
    else:
        builders = {
            "train": lambda: TrainFMRIDataset(os.path.join(root, "train"), target, augment=data_cfg.augment),
            "val": lambda: ValFMRIDataset(os.path.join(root, "val"), target),
        }
    builders["test"] = lambda: TestFMRIDataset(os.path.join(root, "test"), target)

    loaders = {}
    for split in splits:
        dataset = builders[split]()
        is_train = split == "train"
        sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train,
                                      drop_last=is_train)
                   if distributed and split != "test" else None)
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=is_train and sampler is None, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=is_train,
            persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None,
        )
    return loaders
