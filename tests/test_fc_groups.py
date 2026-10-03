import csv
from collections import Counter

import nibabel as nib
import numpy as np
import pytest
from torch.utils.data import DataLoader

from moco.data.fc_groups import FCGroupBatchSampler, FCGroupDataset, split_groups

K = 4
# (subject, session, grade, n_chunks); S3's run 001 exists in two sessions; the Grade 6 scan is too short
SCANS = [("S1", "1", "Grade 1", 9), ("S2", "1", "Grade 1", 4), ("S3", "1", "Grade 2", 8), ("S3", "2", "Grade 2", 6),
         ("S4", "1", "Grade 3", 4), ("S5", "1", "Grade 4", 5), ("S6", "1", "Grade 5", 4), ("S7", "1", "Grade 6", 3)]


@pytest.fixture
def dataset(tmp_path) -> FCGroupDataset:
    rows, stats, splits = [], [], {}
    for subject, session, grade, n_chunks in SCANS:
        for c in range(n_chunks):
            path = tmp_path / f"sub-{subject}_ses-{session}_chunk-{c}.nii.gz"
            nib.save(nib.Nifti1Image(np.full((6, 4, 4, 5), c + 1, dtype=np.float32), np.eye(4)), path)
            rows.append(dict(split="train", grade=grade, task="videos", subject_id=subject, age_group="2mo",
                             session_id=session, run_id="001", chunk_start=str(5 * c), chunk_path=str(path)))
        stats.append(dict(subject_id=subject, session_id=session, run_id="001", task="videos", median=0, scale=1))
        splits[subject] = dict(subject_id=subject, task="videos", split="train")
    for name, table in (("meta.csv", rows), ("stats.csv", stats), ("splits.csv", list(splits.values()))):
        with open(tmp_path / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=table[0].keys())
            writer.writeheader()
            writer.writerows(table)
    return FCGroupDataset("train", str(tmp_path / "meta.csv"), str(tmp_path / "stats.csv"),
                          str(tmp_path / "splits.csv"), clean_grade="Grade 1",
                          grades_b=["Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6"], padded_h=8,
                          chunks_per_example=K)


def _sampler(ds: FCGroupDataset, **kwargs) -> FCGroupBatchSampler:
    args = dict(a_group_sizes=[len(g) for g in ds.A_groups], a_grades=[g[0]["grade"] for g in ds.A_groups],
                b_group_sizes=[len(g) for g in ds.B_groups], k=K, batch_size=2, shuffle=True, seed=0)
    return FCGroupBatchSampler(**{**args, **kwargs})


def test_split_groups_uses_every_full_set():
    sets = split_groups([9, 4, 3], K, np.random.default_rng(0))
    assert Counter(g for g, _ in sets) == {0: 2, 1: 1}  # 9 -> 2 sets, 4 -> 1, 3 -> none
    group0 = [i for g, chunks in sets if g == 0 for i in chunks]
    assert len(group0) == len(set(group0)) == 8  # disjoint, one chunk left over
    assert split_groups([9], K, None) == [(0, (0, 1, 2, 3)), (0, (4, 5, 6, 7))]


def test_groups_follow_scan_and_skip_short_ones(dataset):
    assert [len(g) for g in dataset.A_groups] == [8, 6, 4, 5, 4]  # two S3 sessions kept apart, Grade 6 dropped
    assert len(dataset) == 6  # 2 + 1 + 1 + 1 + 1 sets


def test_item_shape_and_metadata(dataset):
    item = dataset[(0, (1, 3, 5, 7), 0, (0, 2, 4, 6))]
    assert item["A"].shape == item["B"].shape == (K, 5, 8, 4, 4)
    assert item["A_meta"]["chunk_start"].tolist() == [5, 15, 25, 35]
    assert (item["A_meta"]["motion_grade"], item["A_meta"]["session_id"]) == ("Grade 2", "1")
    with pytest.raises(AssertionError):
        dataset[(0, (1, 1, 2, 3), 0, (0, 1, 2, 3))]


def test_epoch_uses_every_set_once(dataset):
    sampler = _sampler(dataset)
    a_sets = [(key[0], key[1]) for batch in sampler for key in batch]
    assert len(a_sets) == len(set(a_sets)) == len(dataset) == len(sampler) * 2
    for g in range(len(dataset.A_groups)):
        chunks = [c for group, cs in a_sets if group == g for c in cs]
        assert len(chunks) == len(set(chunks))  # no chunk twice within an epoch


def test_batches_mix_grades_while_possible(dataset):
    for batch in _sampler(dataset):
        grades = [dataset.A_groups[key[0]][0]["grade"] for key in batch]
        assert len(set(grades)) == len(grades)  # Grade 2 has 3 of 6 sets, so 2-slot batches can always mix


def test_reproducible_per_epoch(dataset):
    sampler = _sampler(dataset)
    first = list(sampler)
    assert first == list(_sampler(dataset))
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_ddp_ranks_equal_and_disjoint(dataset):
    ranks = [list(_sampler(dataset, batch_size=1, rank=r, world_size=2)) for r in range(2)]
    assert len(ranks[0]) == len(ranks[1]) == 3
    assert not {k for b in ranks[0] for k in b} & {k for b in ranks[1] for k in b}


def test_validation_is_fixed_and_complete(dataset):
    val = _sampler(dataset, batch_size=4, shuffle=False)
    batches = list(val)
    assert [len(b) for b in batches] == [4, 2] and len(val) == 2  # partial batch kept
    val.set_epoch(5)
    assert list(val) == batches


def test_dataloader_batch_shapes(dataset):
    batch = next(iter(DataLoader(dataset, batch_sampler=_sampler(dataset))))
    assert batch["A"].shape == batch["B"].shape == (2, K, 5, 8, 4, 4)
    assert batch["A_meta"]["chunk_start"].shape == (2, K)
    assert len(batch["A_meta"]["subject_id"]) == 2
