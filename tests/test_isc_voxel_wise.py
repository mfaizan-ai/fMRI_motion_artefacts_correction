import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize

from moco.evaluation.isc_voxel_wise import (
    bootstrap_counts,
    group_statistics,
    loso_isc,
    network_isc,
    to_nifti,
    voxel_corr,
    zscore_time,
)
from moco.visualization.isc_voxel_plots import plot_networks


def test_zscore_time_standardises_and_leaves_flat_voxels_at_zero():
    x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
    z = zscore_time(x)
    np.testing.assert_allclose(z[0].mean(), 0, atol=1e-12)
    np.testing.assert_allclose(z[0].std(), 1)
    np.testing.assert_array_equal(z[1], 0)


def test_loso_reference_excludes_the_subject_and_averages_its_scans_in_fisher_z():
    rng = np.random.default_rng(0)
    shared = rng.normal(size=(5, 60))  # 5 voxels, 60 time points of stimulus-driven signal
    scans = {s: [zscore_time(shared + rng.normal(scale=1.5, size=(5, 60))) for _ in range(n)]
             for s, n in (("S1", 2), ("S2", 1), ("S3", 1))}
    subjects, subject_z, qc = loso_isc(scans)
    assert subjects == ["S1", "S2", "S3"] and subject_z.shape == (3, 5)
    reference = np.mean([np.mean(scans[s], axis=0) for s in ("S2", "S3")], axis=0)  # built directly, without S1
    direct = np.mean([np.arctanh(voxel_corr(scan, reference)) for scan in scans["S1"]], axis=0)
    np.testing.assert_allclose(subject_z[0], direct, rtol=1e-5)
    assert len(qc) == 4 and (qc.groupby("subject")["n_target_scans"].first() == [2, 1, 1]).all()


def test_bootstrap_counts_are_seeded_and_draw_every_subject_slot():
    a, b = bootstrap_counts(10, 50, seed=3), bootstrap_counts(10, 50, seed=3)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (50, 10) and (a.sum(axis=1) == 10).all()


def test_group_statistics_separates_signal_from_noise_and_masks_non_significant_voxels():
    rng = np.random.default_rng(1)
    subject_z = np.column_stack([rng.normal(0.3, 0.05, 30), rng.normal(0.0, 0.05, 30)]).astype(np.float32)
    stats = group_statistics(subject_z, bootstrap_counts(30, 500, seed=0), ci=0.95, alpha=0.05)
    assert stats["p"][0] == pytest.approx(1 / 501)  # no shifted resample reaches the observed mean
    assert stats["p"][1] > 0.05
    assert stats["ci_low_z"][0] < stats["mean_z"][0] < stats["ci_high_z"][0]
    assert np.isfinite(stats["mean_z_fdr"][0]) and np.isnan(stats["mean_z_fdr"][1])


def test_to_nifti_puts_values_on_the_mask_and_nan_elsewhere():
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, 1, :] = True
    img = to_nifti(np.array([1.0, 2.0, 3.0]), mask, np.eye(4))
    data = img.get_fdata()
    np.testing.assert_array_equal(data[mask], [1, 2, 3])
    assert np.isnan(data[~mask]).all()
    assert to_nifti(np.ones((4, 3)), mask, np.eye(4)).shape == (3, 3, 3, 4)  # (maps, voxels) -> 4D


def test_network_isc_averages_voxels_per_network_then_bootstraps_subjects():
    z = {"raw": np.array([[0.1, 0.3, 0.5], [0.3, 0.5, 0.7]]), "corrected": np.zeros((2, 3))}
    table = network_isc(z, np.array(["Vis", "Vis", "Default"]), ["Vis", "Default"], bootstrap_counts(2, 20, 0),
                        ci=0.95)
    vis_raw = table[(table["network"] == "Vis") & (table["map"] == "raw")].iloc[0]
    assert vis_raw["n_voxels"] == 2 and vis_raw["mean_z"] == pytest.approx(0.3)  # mean of (0.2, 0.4)
    assert len(table) == 4


def test_plot_networks_draws_both_panels_from_a_network_table():
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose("isc_voxel_wise", overrides=["source=unused"])
    rows = [{"network": n, "map": m, "mean_z": 0.1, "ci_low_z": 0.0, "ci_high_z": 0.2}
            for n in cfg.networks for m in ("raw", "corrected", "corrected_minus_raw")]
    assert len(plot_networks(pd.DataFrame(rows), "A", cfg).axes) == 2
