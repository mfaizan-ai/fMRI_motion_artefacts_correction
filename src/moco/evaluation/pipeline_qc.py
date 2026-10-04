"""Pipeline-level QC of raw or denoised whole runs: per-subject metrics, then group QC-FC, FC significance and Q."""
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nilearn.signal import clean
from omegaconf import DictConfig
from scipy import stats

from moco.atlas import SchaeferAtlas, SchaeferAtlasCropped
from moco.data.grade import pad_axis0
from moco.evaluation import qc_metrics as qc
from moco.evaluation.denoise import corrected_path, select_runs
from moco.losses.sequence import pearson_corr_matrix

log = logging.getLogger(__name__)

RAW_SOURCE = "raw"


def output_dir(cfg: DictConfig) -> Path:
    """`qc_root/original_data` for raw runs, `qc_root/<denoised folder name>` otherwise."""
    return Path(cfg.qc_root) / (cfg.raw_output_name if cfg.source == RAW_SOURCE else cfg.source)


def select_qc_runs(cfg: DictConfig) -> pd.DataFrame:
    """First video run of every 2-month subject, with `volume_path` pointing at raw or denoised data.

    A denoised source must hold every one of these runs (found by `corrected_path`), so raw and every
    model are compared on the same subjects.

    Raises:
        FileNotFoundError: If any selected run is missing from the denoised source.
    """
    runs = select_runs(cfg.data.chunk_metadata_csv, "first_2mo")
    if cfg.max_subjects is not None:
        runs = runs.head(cfg.max_subjects)
    if cfg.source == RAW_SOURCE:
        return runs.assign(volume_path=runs["source_volume_path"])
    denoised_dir = Path(cfg.denoised_root) / cfg.source
    paths = [str(corrected_path(Path(p), Path(cfg.source_root), denoised_dir)) for p in runs["source_volume_path"]]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)}/{len(paths)} selected runs missing from {denoised_dir}, "
                                f"e.g. {missing[0]}")
    return runs.assign(volume_path=paths)


def roi_timeseries(func: np.ndarray, atlas: SchaeferAtlas, padded_h: int, high_pass_hz: float,
                   tr_seconds: float) -> np.ndarray:
    """Atlas ROI means of one run, cosine high-pass filtered as in the original evaluation.

    Args:
        func: (X, Y, Z, T) run on the cropped grid.
        atlas: Atlas on the padded grid.
        padded_h: Size axis 0 is zero-padded to, matching the atlas.
        high_pass_hz: Cosine high-pass cut-off.
        tr_seconds: Repetition time.

    Returns:
        (T, R) ROI time series.
    """
    volumes = pad_axis0(torch.from_numpy(func).permute(3, 0, 1, 2), padded_h)  # (T, H, Y, Z)
    series = atlas.extract_roi_timeseries(volumes).numpy()
    return clean(series, detrend=False, standardize=None, filter="cosine", high_pass=high_pass_hz, t_r=tr_seconds)


def subject_metrics(row, atlas: SchaeferAtlas, cfg: DictConfig) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run-level metrics of one subject.

    Returns:
        (record, roi time series (T, R), FC (R, R)).
    """
    func = np.asarray(nib.load(row.volume_path).dataobj, dtype=np.float32)
    mask = func[..., 0] != 0  # runs are brain-masked, so the mask is the nonzero voxels of the first volume
    series = roi_timeseries(func, atlas, cfg.data.spatial_dims[0], cfg.high_pass_hz, row.tr_seconds)
    fc = pearson_corr_matrix(torch.from_numpy(series).float()).numpy()
    q, n_communities = qc.consensus_q(fc, cfg.modularity.gamma, cfg.modularity.repeats, cfg.modularity.seed)
    dvars = qc.dvars_timeseries(func, mask, cfg.dvars.intensity_normalization, cfg.dvars.variance_tol)
    record = {
        "subject_id": row.subject_id, "session_id": row.session_id, "run_id": row.run_id,
        "volume_path": row.volume_path,
        "mean_fd": float(pd.read_csv(row.fd_path)["FramewiseDisplacement"].mean()),
        "Q": q, "n_communities": n_communities,
        "dvars": float(dvars.mean()), "tsnr": qc.tsnr(func, mask, cfg.tsnr_std_floor),
    }
    return record, series, fc


def run_pipeline_qc(cfg: DictConfig) -> None:
    """Compute and save per-subject and group-level QC for one data source.

    Args:
        cfg: Composed `configs/pipeline_qc.yaml`.
    """
    out_dir = output_dir(cfg)
    runs = select_qc_runs(cfg)
    atlas = SchaeferAtlasCropped(cfg.atlas.paths["2mo"], "2mo", cfg.atlas.crop_window["2mo"],
                                 tuple(cfg.data.spatial_dims))
    log.info("pipeline QC of %s: %d subjects -> %s", cfg.source, len(runs), out_dir)

    roi_dir = out_dir / "roi_timeseries"
    roi_dir.mkdir(parents=True, exist_ok=True)
    records, edges = [], []
    for i, row in enumerate(runs.itertuples(index=False), 1):
        record, series, fc = subject_metrics(row, atlas, cfg)
        np.save(roi_dir / f"sub-{row.subject_id}_ses-{row.session_id}_run-{row.run_id}.npy", series)
        records.append(record)
        edges.append(qc.fisher_z(qc.upper_triangle(fc)))
        log.info("[%d/%d] %s Q=%.4f DVARS=%.2f tSNR=%.2f", i, len(runs), row.subject_id, record["Q"],
                 record["dvars"], record["tsnr"])
    subjects = pd.DataFrame(records)
    subjects.to_csv(out_dir / "per_subject.csv", index=False)

    edges = np.stack(edges)  # (S, E)
    mean_fd, n_rois = subjects["mean_fd"].to_numpy(), atlas.n_rois
    qcfc_r, qcfc_p = qc.qc_fc(edges, mean_fd)
    qcfc_sig, _ = qc.fdr(qcfc_p, cfg.edge_alpha)
    distance = qc.roi_distance_matrix(cfg.atlas.paths["2mo"])
    if distance.shape != (n_rois, n_rois):
        raise ValueError(f"distance matrix {distance.shape} does not match {n_rois} ROIs")
    dd_r, dd_p = qc.distance_dependence(qcfc_r, qc.upper_triangle(distance))
    fc_mean_z, fc_p = qc.fc_significance(edges)
    fc_sig, fc_q = qc.fdr(fc_p, cfg.edge_alpha)
    q_fd_r, q_fd_p = stats.pearsonr(subjects["Q"], mean_fd)

    arrays = {"qc_fc_r": qcfc_r, "qc_fc_p": qcfc_p, "fc_mean_z": fc_mean_z, "fc_p": fc_p, "fc_q": fc_q}
    for name, vec in arrays.items():
        np.save(out_dir / f"{name}.npy", qc.edges_to_matrix(vec, n_rois))
    np.save(out_dir / "qc_fc_fdr_significant.npy", qc.edges_to_matrix(qcfc_sig, n_rois, fill=False))
    np.save(out_dir / "fc_significant.npy", qc.edges_to_matrix(fc_sig, n_rois, fill=False))
    np.save(out_dir / "distance_matrix.npy", distance)

    n_edges = edges.shape[1]
    summary = {
        "source": cfg.source, "n_subjects": len(subjects), "n_edges": n_edges,
        "qcfc_n_significant": int(qcfc_sig.sum()), "qcfc_pct_significant": 100 * float(qcfc_sig.mean()),
        "qcfc_median_abs_r": float(np.nanmedian(np.abs(qcfc_r))),
        "qcfc_dd_r": dd_r, "qcfc_dd_p": dd_p,
        "fc_n_significant": int(fc_sig.sum()), "fc_pct_significant": 100 * float(fc_sig.mean()),
        "fc_median_abs_r": float(np.median(np.abs(np.tanh(fc_mean_z)))),
        "Q_mean": float(subjects["Q"].mean()), "Q_sd": float(subjects["Q"].std()),
        "Q_vs_fd_r": float(q_fd_r), "Q_vs_fd_p": float(q_fd_p),
        "dvars_mean": float(subjects["dvars"].mean()), "dvars_sd": float(subjects["dvars"].std()),
        "tsnr_mean": float(subjects["tsnr"].mean()), "tsnr_sd": float(subjects["tsnr"].std()),
        "mean_fd_mean": float(mean_fd.mean()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("QC-FC: %d/%d FDR-significant, median |r| = %.4f, DD r = %.4f; FC significant %d; Q = %.4f",
             summary["qcfc_n_significant"], n_edges, summary["qcfc_median_abs_r"], dd_r,
             summary["fc_n_significant"], summary["Q_mean"])
