import csv

import nibabel as nib
import numpy as np
import pytest
import torch

from moco.data.grade import (
    FMRIUnpairedGradeDataset,
    crop_axis0,
    denormalize_chunk,
    normalize_volume,
    pad_axis0,
)


def test_pad_crop_inverse():
    x = torch.randn(5, 60, 7, 6)
    padded = pad_axis0(x, 64)
    assert padded.shape == (5, 64, 7, 6) and torch.all(padded[:, :2] == 0) and torch.all(padded[:, -2:] == 0)
    assert torch.equal(crop_axis0(padded, 60), x)


def test_normalise_denormalise_roundtrip():
    data = np.random.default_rng(0).uniform(50, 150, size=(6, 5, 4, 5)).astype(np.float32)
    data[0] = 0
    x = torch.from_numpy(normalize_volume(data, 100.0, 10.0)).permute(3, 0, 1, 2).unsqueeze(0)
    back = denormalize_chunk(x, torch.tensor([100.0]), torch.tensor([10.0]), x != 0)
    np.testing.assert_allclose(back[0].permute(1, 2, 3, 0).numpy(), data, rtol=1e-5)


@pytest.fixture
def tiny_grade_dataset(tmp_path):
    rows, splits = [], []
    for i, grade in enumerate(["Grade 1"] * 3 + ["Grade 2"] * 5 + ["Grade 4"] * 2):
        path = tmp_path / f"sub-S{i}_chunk.nii.gz"
        nib.save(nib.Nifti1Image(np.full((6, 4, 4, 5), i + 1, dtype=np.float32), np.eye(4)), path)
        rows.append(dict(split="train", grade=grade, task="videos", subject_id=f"S{i}", age_group="2mo",
                         session_id="1", run_id="001", chunk_start="0", chunk_end="5", chunk_mean_fd="0.1",
                         chunk_max_fd="0.2", chunk_path=str(path), source_volume_path="x"))
        splits.append(dict(subject_id=f"S{i}", task="videos", split="train"))
    for name, table in (("meta.csv", rows), ("splits.csv", splits)):
        with open(tmp_path / name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=table[0].keys())
            writer.writeheader()
            writer.writerows(table)
    with open(tmp_path / "stats.csv", "w") as f:
        f.write("subject_id,session_id,run_id,task,median,scale\n")
        f.writelines(f"S{i},1,001,videos,0,1\n" for i in range(10))
    return dict(chunk_metadata_csv=str(tmp_path / "meta.csv"), run_stats_csv=str(tmp_path / "stats.csv"),
                splits_csv=str(tmp_path / "splits.csv"), clean_grade="Grade 1",
                grades_b=["Grade 2", "Grade 3", "Grade 4"], padded_h=8)


def test_epoch_sampling(tiny_grade_dataset):
    ds = FMRIUnpairedGradeDataset("train", **tiny_grade_dataset)
    assert (ds.A_size, ds.B_size, len(ds)) == (7, 3, 3)  # the smaller domain (B) sets the epoch length
    item = ds[0]
    assert item["A"].shape == (5, 8, 4, 4) and item["B_meta"]["grade"] == "Grade 1"
    ds.set_epoch(1)
    first = [ds[i]["A_paths"] for i in range(len(ds))]
    ds.set_epoch(1)
    assert first == [ds[i]["A_paths"] for i in range(len(ds))]  # reproducible per epoch
    assert len(set(first)) == len(first)  # no repeats within an epoch


def test_full_coverage_visits_every_chunk(tiny_grade_dataset):
    ds = FMRIUnpairedGradeDataset("train", full_coverage=True, **tiny_grade_dataset)
    assert len(ds) == 7
    assert len({ds[i]["A_paths"] for i in range(len(ds))}) == 7
