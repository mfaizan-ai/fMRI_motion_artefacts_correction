import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize

from moco.evaluation.isc import Segment, bootstrap_isc, leave_one_out, representative_segment, roi_networks
from moco.visualization.isc_plots import plot_network_isc, plot_network_isfc


def _segment(session, run, segment_num, roi=None):
    return Segment(session, run, segment_num, roi, None)


def test_representative_segment_prefers_same_session_and_run_then_nearest():
    entries = [_segment(1, 1, 2), _segment(1, 3, 1), _segment(2, 2, 1)]
    assert representative_segment(entries, 1, 1, 1) is entries[0]  # exact session/run beats segment match
    assert representative_segment(entries, 1, 2, 1) is entries[1]  # same session; runs 1 and 3 tie, same segment wins
    assert representative_segment(entries, 3, 2, 1) is entries[2]  # no session 3: same run
    assert representative_segment(entries, 3, 9, 1) is entries[2]  # nothing shared: nearest session


def test_leave_one_out_diagonal_is_the_rowwise_isc_against_the_others_mean():
    rng = np.random.default_rng(0)
    shared = rng.normal(size=(3, 50))  # stimulus-driven signal every subject sees
    series = {f"S{i}": [_segment(1, 1, 1, shared + rng.normal(scale=s, size=(3, 50)))]
              for i, s in enumerate([0.5, 1.0, 2.0])}
    subjects, isfc = leave_one_out(series, "roi")
    assert subjects == ["S0", "S1", "S2"] and isfc.shape == (3, 3, 3)
    mine, others = series["S2"][0].roi, np.mean([series["S0"][0].roi, series["S1"][0].roi], axis=0)
    expected = [np.corrcoef(mine[k], others[k])[0, 1] for k in range(3)]
    np.testing.assert_allclose(np.diagonal(isfc[2]), expected, rtol=1e-6)
    assert np.all(np.diagonal(isfc[0]) > np.diagonal(isfc[2]))  # less noise, higher ISC


def test_leave_one_out_averages_a_subjects_viewings_in_fisher_z():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(2, 40))
    series = {"S1": [_segment(1, 1, 1, base + rng.normal(size=(2, 40))),
                     _segment(1, 2, 2, base + rng.normal(size=(2, 40)))],
              "S2": [_segment(1, 1, 1, base + rng.normal(size=(2, 40)))]}
    _, isfc = leave_one_out(series, "roi")
    others = series["S2"][0].roi  # S2 is S1's only other subject, for both viewings
    r = np.array([[np.corrcoef(e.roi[k], others[k])[0, 1] for k in range(2)] for e in series["S1"]])
    np.testing.assert_allclose(np.diagonal(isfc[0]), np.tanh(np.arctanh(r).mean(axis=0)), rtol=1e-6)


def test_bootstrap_isc_is_seeded_and_separates_signal_from_noise():
    rng = np.random.default_rng(0)
    isc = np.column_stack([rng.normal(0.3, 0.05, 40), rng.normal(0.0, 0.05, 40)])
    a = bootstrap_isc(isc, 2000, 0.95, np.random.default_rng(7))
    b = bootstrap_isc(isc, 2000, 0.95, np.random.default_rng(7))
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert a["p"][0] == pytest.approx(1 / 2001)  # no resample reaches zero: smallest possible p
    assert a["p"][1] > 0.05
    assert np.all((a["ci_low"] <= a["group_isc"]) & (a["group_isc"] <= a["ci_high"]))


def test_roi_networks_maps_label_order_and_rejects_lost_rois(tmp_path):
    names = ["7Networks_LH_Vis_1", "7Networks_LH_Default_1", "7Networks_RH_Vis_2"]
    pd.DataFrame({"ROI Label": [1, 2, 3], "ROI Name": names}).to_csv(tmp_path / "c.csv", index=False)

    class Atlas:
        active_labels, n_rois = [1, 2, 3], 3

    roi_names, network_names, index = roi_networks(str(tmp_path / "c.csv"), Atlas())
    assert roi_names == names and network_names == ["Default", "Vis"]
    np.testing.assert_array_equal(index, [1, 0, 1])
    Atlas.active_labels = [1, 3]
    with pytest.raises(ValueError):
        roi_networks(str(tmp_path / "c.csv"), Atlas())


def test_isc_plots_draw_from_synthetic_results():
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose("isc_network")
    table = pd.DataFrame({"name": ["Default", "Vis"], "group_isc": [0.1, 0.3], "ci_low": [0.0, 0.2],
                          "ci_high": [0.2, 0.4], "significant": [False, True]})
    assert len(plot_network_isc(table, "t", cfg).axes) == 1
    assert len(plot_network_isfc(np.array([[0.1, 0.05], [0.05, 0.3]]), ["Default", "Vis"], "t", cfg).axes) == 2
