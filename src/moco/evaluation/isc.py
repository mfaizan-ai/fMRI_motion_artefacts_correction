"""Leave-one-out inter-subject correlation (ISC) of ROI and Yeo-7 network time series, per video order."""
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from moco.atlas import SchaeferAtlas, SchaeferAtlasCropped
from moco.data.grade import pad_axis0
from moco.evaluation.denoise import check_output_root, corrected_path
from moco.evaluation.pipeline_qc import RAW_SOURCE, output_dir
from moco.evaluation.qc_metrics import fdr, fisher_z

log = logging.getLogger(__name__)


@dataclass
class Segment:
    """One viewing of the order's video: (R, T) ROI and (N, T) network time series."""

    session: int
    run: int
    segment_num: int
    roi: np.ndarray
    network: np.ndarray


def load_segments(cfg: DictConfig, order: str) -> pd.DataFrame:
    """Segments of one order for 2-month subjects, with `volume_path` pointing at raw or denoised runs.

    Raises:
        FileNotFoundError: If any segment's run is missing from the source.
    """
    table = pd.read_csv(cfg.segments_csv)
    table = table[(table["order_label"] == order) & ~table["subject"].str.endswith(cfg.exclude_subject_suffix)]
    # the CSV points at the uncropped derivative; the same subtree exists under source_root
    raw_paths = table["bold_path"].str.replace(cfg.segments_bold_root, cfg.source_root, regex=False)
    if cfg.source == RAW_SOURCE:
        paths = raw_paths
    else:
        denoised_dir = Path(cfg.denoised_root) / cfg.source
        paths = [str(corrected_path(Path(p), Path(cfg.source_root), denoised_dir)) for p in raw_paths]
    segments = pd.DataFrame({
        "subject": table["subject"].to_numpy(), "session": table["session"].astype(int).to_numpy(),
        "run": table["run"].astype(int).to_numpy(), "segment_num": table["segment_num"].astype(int).to_numpy(),
        "video": table["first_video_name"].to_numpy(), "start": table["scan_start_idx"].astype(int).to_numpy(),
        "end": table["scan_end_idx"].astype(int).to_numpy(), "volume_path": list(paths),
    })
    missing = sorted({p for p in segments["volume_path"] if not Path(p).exists()})
    if missing:
        raise FileNotFoundError(f"order {order}: {len(missing)} runs missing from {cfg.source}, e.g. {missing[0]} "
                                "(denoise with runs=all_2mo)")
    return segments


def roi_networks(centroids_csv: str, atlas: SchaeferAtlas) -> tuple[list[str], list[str], np.ndarray]:
    """ROI names, sorted Yeo-7 network names and the (R,) network index of each atlas column.

    Raises:
        ValueError: If the atlas lost ROIs, so its columns no longer line up with the label table.
    """
    labels = pd.read_csv(centroids_csv).set_index("ROI Label")["ROI Name"]
    if atlas.active_labels != labels.index.tolist():
        raise ValueError(f"atlas has {atlas.n_rois} ROIs, label table {len(labels)}")
    roi_names = labels.tolist()
    network_of_roi = [name.split("_")[2] for name in roi_names]  # 7Networks_LH_Vis_1 -> Vis
    network_names = sorted(set(network_of_roi))
    return roi_names, network_names, np.array([network_names.index(n) for n in network_of_roi])


def extract_segments(segments: pd.DataFrame, atlas: SchaeferAtlas, network_index: np.ndarray,
                     padded_h: int) -> dict[str, list[Segment]]:
    """ROI and network time series of every segment, grouped by subject; each run is loaded once.

    Args:
        segments: Output of `load_segments`.
        atlas: Atlas on the padded grid.
        network_index: (R,) network of each ROI column.
        padded_h: Size axis 0 is zero-padded to, matching the atlas.

    Returns:
        Subject -> its segments.
    """
    series: dict[str, list[Segment]] = {}
    n_networks = int(network_index.max()) + 1
    for path, run_segments in segments.groupby("volume_path", sort=False):
        func = np.asarray(nib.load(path).dataobj, dtype=np.float32)  # (X, Y, Z, T)
        for row in run_segments.itertuples(index=False):
            volumes = pad_axis0(torch.from_numpy(func[..., row.start:row.end]).permute(3, 0, 1, 2), padded_h)
            roi = atlas.extract_roi_timeseries(volumes).numpy().T  # (R, T)
            # network signal = unweighted mean of its ROI signals, as in the original analysis
            network = np.stack([roi[network_index == k].mean(axis=0) for k in range(n_networks)])
            series.setdefault(row.subject, []).append(Segment(row.session, row.run, row.segment_num, roi, network))
    return series


def representative_segment(entries: list[Segment], session: int, run: int, segment_num: int) -> Segment:
    """The one segment that stands for a subject in another subject's leave-one-out mean.

    A subject never contributes an average of its own viewings. Priority: same session and run, then same
    session and nearest run, then same run and nearest session, then nearest overall; ties prefer the same
    segment_num, then the lowest (session, run, segment_num).
    """
    def other(e: Segment) -> bool:
        return e.segment_num != segment_num

    exact = [e for e in entries if e.session == session and e.run == run]
    if exact:
        return min(exact, key=lambda e: (other(e), e.segment_num))
    same_session = [e for e in entries if e.session == session]
    if same_session:
        return min(same_session, key=lambda e: (abs(e.run - run), other(e), e.run, e.segment_num))
    same_run = [e for e in entries if e.run == run]
    if same_run:
        return min(same_run, key=lambda e: (abs(e.session - session), other(e), e.session, e.segment_num))
    return min(entries, key=lambda e: (abs(e.session - session), abs(e.run - run), other(e), e.session, e.run,
                                       e.segment_num))


def pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(K, T) x (K, T) -> (K, K) correlation of every row of `a` with every row of `b`."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a @ b.T) / np.outer(np.sqrt((a ** 2).sum(axis=1)), np.sqrt((b ** 2).sum(axis=1)))


def natural_key(text: str) -> list:
    """'ICC9' before 'ICC10'."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def leave_one_out(series: dict[str, list[Segment]], key: str) -> tuple[list[str], np.ndarray]:
    """Each subject's segments correlated with the mean of the other subjects' representative segments.

    Args:
        series: Subject -> segments.
        key: "roi" or "network".

    Returns:
        (subjects, ISFC (S, K, K)): row k of subject s correlates its signal k with the others' mean of
        every signal; the diagonal is the ISC. Fisher-z averaged over the subject's own segments, so the
        diagonal equals the original analysis's row-wise ISC.
    """
    subjects = sorted(series, key=natural_key)
    isfc = []
    for subject in subjects:
        r = []
        for e in series[subject]:
            others = [getattr(representative_segment(series[s], e.session, e.run, e.segment_num), key)
                      for s in subjects if s != subject]
            r.append(pearson(getattr(e, key), np.mean(others, axis=0)))
        isfc.append(np.tanh(fisher_z(np.stack(r)).mean(axis=0)))
    return subjects, np.stack(isfc)


def bootstrap_isc(isc: np.ndarray, n_boot: int, ci: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Subject-wise bootstrap of the group ISC (Chen et al. 2016, as in BrainIAK `bootstrap_isc`).

    Args:
        isc: (S, K) per-subject leave-one-out ISC.
        n_boot: Number of resamples of subjects with replacement.
        ci: Confidence-interval coverage, e.g. 0.95.
        rng: Seeded generator.

    Returns:
        group_isc, ci_low, ci_high and two-sided p, each (K,). The group ISC is tanh of the mean Fisher-z.
    """
    z = fisher_z(isc)
    observed = z.mean(axis=0)
    resamples = rng.integers(0, len(z), size=(n_boot, len(z)))
    boot = np.stack([z[idx].mean(axis=0) for idx in resamples])  # (n_boot, K)
    # null = bootstrap distribution shifted to zero; p counts resamples at least as far from it as observed
    p = (1 + (np.abs(boot - observed) >= np.abs(observed)).sum(axis=0)) / (1 + n_boot)
    low, high = np.tanh(np.percentile(boot, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2], axis=0))
    return {"group_isc": np.tanh(observed), "ci_low": low, "ci_high": high, "p": p}


def group_table(names: list[str], isc: np.ndarray, cfg: DictConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Bootstrap group ISC per ROI or network, with BH-FDR across them."""
    stats = bootstrap_isc(isc, cfg.n_boot, cfg.ci, rng)
    significant, q = fdr(stats["p"], cfg.alpha)
    return pd.DataFrame({"name": names, **stats, "q": q, "significant": significant})


def write_manifest(order_dir: Path, order: str, segments: pd.DataFrame, subjects: list[str],
                   cfg: DictConfig) -> None:
    """Describe how this order's ISC was made in order_dir/manifest.json."""
    manifest = {
        "source": cfg.source, "order": order, "video": segments["video"].iloc[0],
        "segments_csv": str(cfg.segments_csv),
        "segments_csv_sha256": hashlib.sha256(Path(cfg.segments_csv).read_bytes()).hexdigest(),
        "n_subjects": len(subjects), "n_segments": len(segments), "n_runs": segments["volume_path"].nunique(),
        "n_boot": cfg.n_boot, "ci": cfg.ci, "alpha": cfg.alpha, "seed": cfg.seed,
    }
    (order_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run_order(order: str, atlas: SchaeferAtlas, roi_names: list[str], network_names: list[str],
              network_index: np.ndarray, cfg: DictConfig) -> dict:
    """ISC, ISFC and bootstrap statistics of one order, written to `<output>/order_<order>/`.

    Returns:
        Results needed for plotting: network group table, group network ISFC and subject count.
    """
    order_dir = output_dir(cfg) / cfg.output_subdir / f"order_{order}"
    check_output_root(order_dir, cfg.overwrite)
    order_dir.mkdir(parents=True, exist_ok=True)
    segments = load_segments(cfg, order)
    series = extract_segments(segments, atlas, network_index, cfg.data.spatial_dims[0])
    log.info("order %s (%s): %d segments, %d subjects", order, segments["video"].iloc[0], len(segments),
             len(series))

    rng = np.random.default_rng(cfg.seed)  # per order, so one order alone gives the same numbers
    tables = {}
    for key, names in (("network", network_names), ("roi", roi_names)):
        subjects, isfc = leave_one_out(series, key)
        isc = np.diagonal(isfc, axis1=1, axis2=2)  # (S, K)
        pd.DataFrame(isc, index=pd.Index(subjects, name="subject"), columns=names).to_csv(
            order_dir / f"{key}_isc_per_subject.csv")
        tables[key] = group_table(names, isc, cfg, rng)
        tables[key].to_csv(order_dir / f"{key}_isc_group.csv", index=False)
        log.info("order %s %s ISC: %d/%d FDR-significant, median group ISC %.4f", order, key,
                 int(tables[key]["significant"].sum()), len(names), tables[key]["group_isc"].median())
        if key == "network":
            np.save(order_dir / "network_isfc_per_subject.npy", isfc)
            network_isfc = np.tanh(fisher_z(isfc).mean(axis=0))
            network_isfc = (network_isfc + network_isfc.T) / 2  # LOO ISFC is asymmetric; keep the symmetric part
            pd.DataFrame(network_isfc, index=network_names, columns=network_names).to_csv(
                order_dir / "network_isfc_group.csv")
    segments.to_csv(order_dir / "segments.csv", index=False)
    write_manifest(order_dir, order, segments, subjects, cfg)
    return {"order_dir": order_dir, "video": segments["video"].iloc[0], "n_subjects": len(subjects),
            "network": tables["network"], "isfc": network_isfc}


def selected_orders(cfg: DictConfig) -> list[str]:
    """cfg.order alone, or every order in the segments CSV."""
    if cfg.order is not None:
        return [str(cfg.order)]
    return sorted(pd.read_csv(cfg.segments_csv, usecols=["order_label"])["order_label"].unique())


def build_atlas(cfg: DictConfig) -> tuple[SchaeferAtlas, list[str], list[str], np.ndarray]:
    """2-month cropped atlas on the padded grid, plus ROI names, network names and ROI -> network index."""
    atlas = SchaeferAtlasCropped(cfg.atlas.paths["2mo"], "2mo", cfg.atlas.crop_window["2mo"],
                                 tuple(cfg.data.spatial_dims))
    return atlas, *roi_networks(cfg.centroids_csv, atlas)
