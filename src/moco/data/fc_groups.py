"""K-chunk examples from one scan and one grade, for FC estimated over K*T volumes.

Each example stacks K chunks of the same subject, session, run, task and grade (not necessarily
consecutive), so their ROI time series can be pooled into one FC estimate. They are not a continuous
sequence: temporal losses must stay within each T-volume chunk.
"""
from collections import defaultdict

import nibabel as nib
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from moco.data.grade import load_chunk_rows, load_run_stats, normalize_volume, pad_axis0, run_key

GROUP_FIELDS = ("subject_id", "session_id", "run_id", "task", "grade")
ChunkSet = tuple[int, tuple[int, ...]]  # (group index, chunk indices within the group)


def build_groups(rows: list[dict], min_chunks: int) -> list[list[dict]]:
    """Chunk rows grouped by scan and grade, keeping groups with at least min_chunks, in file order."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[f] for f in GROUP_FIELDS)].append(row)
    return [g for g in groups.values() if len(g) >= min_chunks]


def split_groups(group_sizes: list[int], k: int, rng: np.random.Generator | None) -> list[ChunkSet]:
    """Every non-overlapping K-chunk set of every group: random sets if rng is given, else in order.

    A group of n chunks gives n // k sets; which n % k chunks are left out changes with the rng."""
    sets = []
    for g, n in enumerate(group_sizes):
        order = rng.permutation(n) if rng is not None else np.arange(n)
        sets += [(g, tuple(sorted(int(i) for i in order[s:s + k]))) for s in range(0, n - n % k, k)]
    return sets


class FCGroupDataset(Dataset):
    """Unpaired K-chunk examples: A from one Grade 2-6 group, B from one clean-grade group.

    Indexed by keys (a_group, a_chunks, b_group, b_chunks) from FCGroupBatchSampler.
    Items: {"A", "B"}: (K, T, padded_h, W, D), {"A_meta", "B_meta"} with chunk_start: (K,).
    """

    def __init__(self, split: str, chunk_metadata_csv: str, run_stats_csv: str, splits_csv: str,
                 clean_grade: str, grades_b: list[str], padded_h: int, chunks_per_example: int,
                 task: str | None = None):
        super().__init__()
        self.k = chunks_per_example
        self.padded_h = padded_h
        self.run_stats = load_run_stats(run_stats_csv)

        rows_by_grade = load_chunk_rows(chunk_metadata_csv, splits_csv, split, task)
        self.A_groups = build_groups([r for g in grades_b for r in rows_by_grade.get(g, [])], self.k)
        self.B_groups = build_groups(rows_by_grade.get(clean_grade, []), self.k)
        if not self.A_groups or not self.B_groups:
            raise ValueError(f"split={split!r}: {len(self.A_groups)} A and {len(self.B_groups)} B groups "
                             f"with >= {self.k} chunks, both must be > 0")
        missing = {run_key(g[0]) for g in self.A_groups + self.B_groups} - self.run_stats.keys()
        if missing:
            raise ValueError(f"{len(missing)} runs have no normalisation stats, e.g. {next(iter(missing))}")

    def __len__(self) -> int:
        return sum(len(g) // self.k for g in self.A_groups)

    def _load(self, row: dict) -> Tensor:
        median, scale = self.run_stats[run_key(row)]
        data = np.asarray(nib.load(row["chunk_path"]).dataobj, dtype=np.float32)  # (H, W, D, T)
        x = torch.from_numpy(normalize_volume(data, median, scale)).permute(3, 0, 1, 2)
        return pad_axis0(x, self.padded_h)

    def _example(self, group: list[dict], chunk_indices: tuple[int, ...]) -> tuple[Tensor, dict]:
        chosen = [group[i] for i in chunk_indices]
        first = chosen[0]
        assert len(set(chunk_indices)) == self.k, f"expected {self.k} distinct chunks, got {chunk_indices}"
        assert all(tuple(r[f] for f in GROUP_FIELDS) == tuple(first[f] for f in GROUP_FIELDS) for r in chosen), \
            "chunks of one example must share subject, session, run, task and grade"
        median, scale = self.run_stats[run_key(first)]
        meta = dict(
            subject_id=first["subject_id"], session_id=first["session_id"], run_id=first["run_id"],
            state=first["task"], motion_grade=first["grade"], age_group=first["age_group"],
            chunk_start=torch.tensor([int(r["chunk_start"]) for r in chosen]),
            median=median, scale=scale,
        )
        return torch.stack([self._load(r) for r in chosen]), meta

    def __getitem__(self, key: tuple[int, tuple[int, ...], int, tuple[int, ...]]) -> dict:
        a_group, a_chunks, b_group, b_chunks = key
        a, a_meta = self._example(self.A_groups[a_group], a_chunks)
        b, b_meta = self._example(self.B_groups[b_group], b_chunks)
        return {"A": a, "B": b, "A_meta": a_meta, "B_meta": b_meta}


class FCGroupBatchSampler(Sampler):
    """Batches of dataset keys that use every K-chunk set of every A group once per epoch.

    shuffle (training): groups are re-split into random sets each epoch (seed + epoch); each batch takes
    one set from each of the grades with the most sets left (ties random), so grades repeat within a batch
    only when unavoidable and common grades are spread over the epoch; B sets
    are reshuffled and reused when fewer than A sets; the last incomplete batch is dropped and, under DDP,
    each rank takes every world_size-th batch. Without shuffle (validation): fixed in-order sets, B cycled
    in order, nothing dropped.
    """

    def __init__(self, a_group_sizes: list[int], a_grades: list[str], b_group_sizes: list[int], k: int,
                 batch_size: int, shuffle: bool, seed: int, rank: int = 0, world_size: int = 1):
        self.a_group_sizes, self.a_grades, self.b_group_sizes = a_group_sizes, a_grades, b_group_sizes
        self.k, self.batch_size, self.shuffle, self.seed = k, batch_size, shuffle, seed
        self.rank, self.world_size = rank, world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _grade_mixed_order(self, sets: list[ChunkSet], rng: np.random.Generator) -> list[ChunkSet]:
        queues: dict[str, list[ChunkSet]] = defaultdict(list)
        for i in rng.permutation(len(sets)):
            queues[self.a_grades[sets[i][0]]].append(sets[i])
        order: list[ChunkSet] = []
        while any(queues.values()):
            tie_break = dict(zip(sorted(queues), rng.random(len(queues)), strict=True))
            in_batch: set[str] = set()
            for _ in range(self.batch_size):
                left = [g for g, q in queues.items() if q]
                if not left:
                    break
                candidates = [g for g in left if g not in in_batch] or left
                # most sets left first keeps grade counts level, which is what lets later batches still mix
                grade = min(candidates, key=lambda g: (-len(queues[g]), tie_break[g]))
                in_batch.add(grade)
                order.append(queues[grade].pop())
        return order


    def _batches(self) -> list[list[tuple]]:
        rng = np.random.default_rng([self.seed, self.epoch]) if self.shuffle else None
        a_sets = split_groups(self.a_group_sizes, self.k, rng)
        b_sets = split_groups(self.b_group_sizes, self.k, rng)
        if self.shuffle:
            a_sets = self._grade_mixed_order(a_sets, rng)
            n_passes = -(-len(a_sets) // len(b_sets))
            b_sets = [b_sets[i] for _ in range(n_passes) for i in rng.permutation(len(b_sets))]
        else:
            b_sets = [b_sets[i % len(b_sets)] for i in range(len(a_sets))]
        keys = [(*a, *b) for a, b in zip(a_sets, b_sets[: len(a_sets)], strict=True)]
        if self.shuffle:
            # drop the last partial batch; every rank must see the same number of batches or DDP hangs
            n_batches = len(keys) // self.batch_size // self.world_size * self.world_size
            keys = keys[: n_batches * self.batch_size]
        batches = [keys[i:i + self.batch_size] for i in range(0, len(keys), self.batch_size)]
        return batches[self.rank::self.world_size]

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        n_sets = sum(n // self.k for n in self.a_group_sizes)
        if self.shuffle:
            return n_sets // self.batch_size // self.world_size
        return -(-n_sets // self.batch_size)
