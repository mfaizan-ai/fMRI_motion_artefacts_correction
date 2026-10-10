import json

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize
from matplotlib.colors import to_hex

from moco.evaluation.qc_metrics import edges_to_matrix
from moco.visualization.pipeline_qc_plots import plot_pipeline_qc, top_edges
from moco.visualization.test_visualization import bootstrap_median_ci, plot_test_metrics


def test_bootstrap_median_ci_brackets_median_and_is_reproducible():
    values = np.random.default_rng(0).normal(10, 2, size=200)
    ci_a = bootstrap_median_ci(values, 500, 0.95, np.random.default_rng(1))
    ci_b = bootstrap_median_ci(values, 500, 0.95, np.random.default_rng(1))
    assert ci_a == ci_b
    assert ci_a[0] < np.median(values) < ci_a[1]


def test_plot_test_metrics_draws_six_panels_in_grade_order():
    rng = np.random.default_rng(0)
    grades = ["Grade 10", "Grade 2", "Grade 1"]  # string sort would put "Grade 10" before "Grade 2"
    rows = {"grade": np.repeat(grades, 20)}
    for metric in ("tsnr", "dvars", "gs_std"):
        rows |= {f"{metric}_{side}": rng.uniform(1, 5, 60) for side in ("input", "corrected", "improvement")}
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose("plot_test", overrides=["input_dir=unused", "n_boot=50"])
    fig = plot_test_metrics(pd.DataFrame(rows), cfg)
    assert len(fig.axes) == 6
    assert [t.get_text().split("\n")[0] for t in fig.axes[0].get_xticklabels()] == ["Grade 1", "Grade 2", "Grade 10"]


def test_top_edges_keeps_the_strongest_share_of_significant_edges():
    values = edges_to_matrix(np.array([0.1, -0.9, 0.5, 0.3, 2.0, -0.2]), 4, fill=0.0)
    significant = edges_to_matrix(np.array([True, True, True, True, False, True]), 4, fill=False)
    keep, threshold = top_edges(values, significant, top_percent=20)
    # 2.0 is not significant, so only |-0.9| clears the 80th percentile of the significant |values|
    assert threshold == pytest.approx(np.percentile([0.1, 0.9, 0.5, 0.3, 0.2], 80))
    assert keep.sum() == 2 and keep[0, 2]


def test_plot_pipeline_qc_writes_every_figure_from_synthetic_results(tmp_path):
    rng = np.random.default_rng(0)
    n_rois, n_edges = 6, 15
    networks = ["Vis", "Vis", "SomMot", "Default", "Default", "Cont"]
    pd.DataFrame({"ROI Name": [f"7Networks_LH_{n}_{i}" for i, n in enumerate(networks)],
                  "R": rng.uniform(-60, 60, n_rois), "A": rng.uniform(-90, 60, n_rois),
                  "S": rng.uniform(-30, 70, n_rois)}).to_csv(tmp_path / "centroids.csv", index=False)
    for name in ("raw", "model"):
        folder = tmp_path / name
        folder.mkdir()
        sig = edges_to_matrix(rng.random(n_edges) < 0.5, n_rois, fill=False)
        for key, matrix in {"qc_fc_r": edges_to_matrix(rng.uniform(-0.5, 0.5, n_edges), n_rois),
                            "qc_fc_fdr_significant": sig, "fc_significant": sig,
                            "distance_matrix": edges_to_matrix(rng.uniform(10, 100, n_edges), n_rois, fill=0.0),
                            "fc_mean_z": edges_to_matrix(rng.uniform(-1, 1, n_edges), n_rois)}.items():
            np.save(folder / f"{key}.npy", matrix)
        pd.DataFrame({"subject_id": ["a", "b", "c"], "mean_fd": [0.2, 0.8, 1.5],
                      "Q": rng.uniform(0.1, 0.3, 3)}).to_csv(folder / "per_subject.csv", index=False)
        (folder / "summary.json").write_text(json.dumps(
            {"qcfc_median_abs_r": 0.1, "qcfc_pct_significant": 5.0, "qcfc_n_significant": int(sig.sum() // 2),
             "qcfc_dd_r": -0.1, "Q_mean": 0.2, "Q_vs_fd_r": 0.1, "Q_vs_fd_p": 0.5, "fc_pct_significant": 90.0,
             "fc_n_significant": int(sig.sum() // 2)}))
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose("plot_pipeline_qc", overrides=[f"qc_root={tmp_path}", f"centroids_csv={tmp_path}/centroids.csv",
                                                     "fc_top_percent=50", "distribution.kde_points=20",
                                                     "distance.bins=10", "distance.smooth_sigma=1"])
    cfg.sources, cfg.labels, cfg.colors = ["raw", "model"], ["Raw", "Model"], ["#7F7F7F", "#0072B2"]
    figures = plot_pipeline_qc(cfg)
    assert set(figures) == {"qc_fc_distribution", "qc_fc_matrix", "fc_matrix", "qc_fc_distance_dependence",
                            "qc_fc_significant_edges", "modularity_q", "fc_significant_edges"}
    *panels, colorbar = figures["qc_fc_matrix"].axes  # colour bar spans exactly the last matrix
    assert colorbar.get_position().height == pytest.approx(panels[-1].get_position().height)
    assert len(figures["qc_fc_distribution"].axes) == 2  # one panel per source, side by side
    box_faces = [to_hex(box.get_facecolor(), keep_alpha=False) for box in figures["modularity_q"].axes[0].patches]
    assert box_faces == ["#7f7f7f", "#0072b2"]  # one colour per source
    
