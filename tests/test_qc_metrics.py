import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from moco.evaluation import qc_metrics as qc
from moco.evaluation.pipeline_qc import output_dir, select_qc_runs


def test_qc_fc_flags_only_the_motion_driven_edge():
    rng = np.random.default_rng(0)
    mean_fd = rng.uniform(0.1, 2.0, size=60)
    edges = rng.normal(size=(60, 50))
    edges[:, 7] = 0.5 * mean_fd + rng.normal(scale=0.05, size=60)
    r, p = qc.qc_fc(edges, mean_fd)
    significant, _ = qc.fdr(p, alpha=0.05)
    assert r[7] > 0.9
    assert significant[7] and significant.sum() <= 3


def test_qc_fc_rejects_constant_fd():
    with pytest.raises(ValueError):
        qc.qc_fc(np.zeros((5, 3)), np.ones(5))


def test_fdr_leaves_non_finite_p_unsignificant():
    significant, q = qc.fdr(np.array([1e-9, np.nan, 0.9]), alpha=0.05)
    assert significant.tolist() == [True, False, False]
    assert np.isnan(q[1])


def test_edges_to_matrix_inverts_upper_triangle():
    matrix = qc.edges_to_matrix(np.arange(6, dtype=float), 4)
    np.testing.assert_array_equal(matrix, matrix.T)
    np.testing.assert_array_equal(qc.upper_triangle(matrix), np.arange(6))
    assert not qc.edges_to_matrix(np.ones(6, dtype=bool), 4, fill=False).diagonal().any()


def test_dvars_and_tsnr_on_known_signals():
    t = np.arange(10, dtype=np.float32)
    func = np.ones((2, 2, 1, 10), dtype=np.float32) * 100
    func[0, 0, 0] += t  # ramp: constant frame-to-frame step
    func[1, 1, 0] = 0  # outside the brain
    mask = func[..., 0] != 0
    dvars = qc.dvars_timeseries(func, mask, intensity_normalization=0, variance_tol=1e-7)
    # only the ramp voxel has variance; constant voxels are dropped before differencing
    np.testing.assert_allclose(dvars, np.ones(9))
    expected = (100 + t.mean()) / t.std() / 3  # constant voxels have std 0 -> tSNR 0
    assert qc.tsnr(func, mask, std_floor=1e-3) == pytest.approx(expected, rel=1e-5)


def test_modularity_prefers_the_true_two_block_partition():
    rng = np.random.default_rng(0)
    truth = np.repeat([0, 1], 10)
    fc = np.where(truth[:, None] == truth[None, :], 0.6, -0.2) + rng.normal(scale=0.02, size=(20, 20))
    fc = (fc + fc.T) / 2
    assert qc.signed_asymmetric_q(fc, truth, 1.0) > qc.signed_asymmetric_q(fc, rng.permutation(truth), 1.0)
    q, n_communities = qc.consensus_q(fc, gamma=1.0, repeats=5, seed=0)
    assert n_communities == 2
    assert q == pytest.approx(qc.signed_asymmetric_q(fc, truth, 1.0))


def _qc_cfg(tmp_path, source):
    meta = pd.DataFrame({
        "task": "videos", "age_group": "2mo", "subject_id": ["S1", "S1", "S2"], "session_id": "1",
        "run_id": ["001", "002", "001"], "fd_path": "fd.txt", "tr_seconds": 0.61,
        "source_volume_path": [str(tmp_path / "src" / f"sub-{f}" / "bold.nii.gz") for f in ("S1r1", "S1r2", "S2r1")],
    })
    meta.to_csv(tmp_path / "meta.csv", index=False)
    return OmegaConf.create({"source": source, "qc_root": str(tmp_path / "qc"), "raw_output_name": "original_data",
                             "denoised_root": str(tmp_path), "source_root": str(tmp_path / "src"),
                             "max_subjects": None,
                             "data": {"chunk_metadata_csv": str(tmp_path / "meta.csv")}})


def test_raw_source_uses_first_run_per_subject(tmp_path):
    cfg = _qc_cfg(tmp_path, "raw")
    runs = select_qc_runs(cfg)
    assert runs["run_id"].tolist() == ["001", "001"]
    assert (runs["volume_path"] == runs["source_volume_path"]).all()
    assert output_dir(cfg).name == "original_data"


def test_denoised_source_must_cover_the_same_runs(tmp_path):
    cfg = _qc_cfg(tmp_path, "model_a")
    corrected = tmp_path / "model_a" / "sub-S1r1" / "bold_corrected.nii.gz"
    corrected.parent.mkdir(parents=True)
    corrected.touch()
    with pytest.raises(FileNotFoundError, match="1/2 selected runs missing"):
        select_qc_runs(cfg)
    cfg.max_subjects = 1
    assert select_qc_runs(cfg)["volume_path"].tolist() == [str(corrected)]
    assert output_dir(cfg).name == "model_a"
