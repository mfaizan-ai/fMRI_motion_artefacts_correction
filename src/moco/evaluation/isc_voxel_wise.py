"""Voxel-wise leave-one-subject-out ISC for one video order, raw vs one denoising model, with paired bootstrap maps."""
import hashlib
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from omegaconf import DictConfig, OmegaConf

from moco.evaluation.denoise import check_output_root
from moco.evaluation.isc import load_segments
from moco.evaluation.pipeline_qc import RAW_SOURCE
from moco.evaluation.qc_metrics import fdr, fisher_z

log = logging.getLogger(__name__)

DATASETS = ("raw", "corrected")
DIFF = "corrected_minus_raw"
GROUP_MAPS = ("mean_z", "ci_low_z", "ci_high_z", "q_fdr", "mean_z_fdr")
SEGMENT_KEY = ["subject", "session", "run", "segment_num"]


def paired_segments(cfg: DictConfig, order: str) -> pd.DataFrame:
    """Segments of one order with a raw (`path_raw`) and a denoised (`path_corrected`) run path.

    Raises:
        ValueError: If a segment is not exactly cfg.segment_length volumes long.
    """
    raw = load_segments(OmegaConf.merge(cfg, {"source": RAW_SOURCE}), order)
    corrected = load_segments(cfg, order)
    segments = raw.rename(columns={"volume_path": "path_raw"}).merge(
        corrected[[*SEGMENT_KEY, "volume_path"]].rename(columns={"volume_path": "path_corrected"}), on=SEGMENT_KEY)
    if (segments["end"] - segments["start"] != cfg.segment_length).any():
        raise ValueError(f"order {order}: segments must all be {cfg.segment_length} volumes")
    return segments.sort_values(SEGMENT_KEY).reset_index(drop=True)


def check_headers(segments: pd.DataFrame) -> None:
    """Every run of both datasets is on one grid and affine, and long enough for its segment.

    Raises:
        ValueError: On any mismatch, since voxel-wise averaging across subjects needs identical grids.
    """
    grids, affines = set(), set()
    for row in segments.itertuples(index=False):
        for path in (row.path_raw, row.path_corrected):
            img = nib.load(path)
            grids.add(img.shape[:3])
            affines.add(tuple(np.round(img.affine, 3).ravel()))
            if img.shape[3] < row.end:
                raise ValueError(f"{path} has {img.shape[3]} volumes, segment ends at {row.end}")
    if len(grids) != 1 or len(affines) != 1:
        raise ValueError(f"runs are not on one grid: {grids}")


def mask_on_grid(mask_path: str, reference_path: str) -> nib.Nifti1Image:
    """Template mask resampled (nearest) onto the data grid.

    Raises:
        ValueError: If the data grid cuts off part of the mask; same 2 mm voxels, so it must be an exact shift.
    """
    template = nib.load(mask_path)
    mask_img = resample_to_img(template, nib.load(reference_path), interpolation="nearest", force_resample=True,
                               copy_header=True)
    if int(mask_img.get_fdata().sum()) != int(template.get_fdata().sum()):
        raise ValueError("the data grid cuts off part of the mask")
    return mask_img


def zscore_time(x: np.ndarray) -> np.ndarray:
    """(voxels, T) -> per-voxel z-score over T; flat voxels become 0."""
    std = x.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (x - x.mean(axis=1, keepdims=True)) / std


def voxel_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(voxels, T) x (voxels, T) -> (voxels,) Pearson r per voxel."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))


def load_dataset(segments: pd.DataFrame, path_column: str, mask: np.ndarray) -> tuple[dict, np.ndarray]:
    """Cut, mask and z-score every segment of one dataset.

    Returns:
        subject -> list of (voxels, T) z-scored scans, and (voxels,) True where no scan is flat.
    """
    scans, valid = {}, np.ones(int(mask.sum()), dtype=bool)
    for i, row in enumerate(segments.itertuples(index=False), 1):
        data = np.asarray(nib.load(getattr(row, path_column)).dataobj[..., row.start:row.end], dtype=np.float32)[mask]
        valid &= data.std(axis=1) > 0
        scans.setdefault(row.subject, []).append(zscore_time(data))
        if i % 25 == 0 or i == len(segments):
            log.info("%s: loaded %d/%d segments", path_column, i, len(segments))
    return scans, valid


def loso_isc(scans: dict[str, list[np.ndarray]]) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    """Leave-one-subject-out ISC per voxel.

    The reference for a subject is the equal-weight mean of every other subject's representative (the mean of
    its z-scored scans). Each of the subject's own scans is correlated with it and the maps are Fisher-z averaged.

    Args:
        scans: Subject -> list of (voxels, T) z-scored scans, all on the same voxels and T.

    Returns:
        Sorted subjects, (subjects, voxels) Fisher-z ISC, and one QC row per scan.
    """
    subjects = sorted(scans)
    representatives = {s: np.mean(scans[s], axis=0) for s in subjects}
    total = np.sum([representatives[s] for s in subjects], axis=0)
    subject_z, qc = [], []
    for subject in subjects:
        # subtracting the subject's representative removes all of its scans from the reference
        reference = (total - representatives[subject]) / (len(subjects) - 1)
        with np.errstate(invalid="ignore", divide="ignore"):  # flat voxels give NaN; the common mask drops them
            scan_r = np.stack([voxel_corr(scan, reference) for scan in scans[subject]])
        subject_z.append(fisher_z(scan_r).mean(axis=0))
        qc += [{"subject": subject, "scan": k, "n_target_scans": len(scan_r), "median_r": float(np.nanmedian(r)),
                "n_reference_subjects": len(subjects) - 1} for k, r in enumerate(scan_r)]
    return subjects, np.stack(subject_z).astype(np.float32), pd.DataFrame(qc)


def bootstrap_counts(n_subjects: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_subjects) times each subject is drawn; shared by raw, corrected and their paired difference."""
    draws = np.random.default_rng(seed).integers(0, n_subjects, size=(n_boot, n_subjects))
    return np.stack([np.bincount(d, minlength=n_subjects) for d in draws]).astype(np.float32)


def group_statistics(subject_z: np.ndarray, counts: np.ndarray, ci: float, alpha: float) -> dict[str, np.ndarray]:
    """Group mean Fisher z per voxel with percentile CI, two-sided bootstrap p and BH-FDR.

    Args:
        subject_z: (subjects, voxels).
        counts: Output of `bootstrap_counts`.
        ci: CI coverage, e.g. 0.95.
        alpha: FDR level.

    Returns:
        (voxels,) maps: mean_z, ci_low_z, ci_high_z, p, q_fdr, and mean_z_fdr (NaN where not significant).
    """
    boot = counts @ subject_z / subject_z.shape[0]  # (n_boot, voxels) bootstrap group means
    low, high = np.percentile(boot, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2], axis=0)
    # sign test on the bootstrap means; the +1 keeps p > 0, so the smallest p is 2 / (n_boot + 1)
    tail = np.minimum((boot <= 0).sum(axis=0), (boot >= 0).sum(axis=0))
    p = np.minimum(1.0, 2 * (tail + 1) / (len(boot) + 1))
    significant, q = fdr(p, alpha)
    mean_z = subject_z.mean(axis=0)
    return {"mean_z": mean_z, "ci_low_z": low, "ci_high_z": high, "p": p, "q_fdr": q,
            "mean_z_fdr": np.where(significant, mean_z, np.nan)}


def group_summary(stats: dict[str, dict[str, np.ndarray]], alpha: float) -> pd.DataFrame:
    """One row per map: median group ISC, share of voxels with CI above/below 0, FDR-significant voxel counts."""
    rows = []
    for name, s in stats.items():
        significant = s["q_fdr"] < alpha
        rows.append({"map": name, "n_voxels": len(s["mean_z"]), "median_mean_z": float(np.median(s["mean_z"])),
                     "median_r": float(np.tanh(np.median(s["mean_z"]))),
                     "pct_ci_above_0": 100 * float(np.mean(s["ci_low_z"] > 0)),
                     "pct_ci_below_0": 100 * float(np.mean(s["ci_high_z"] < 0)),
                     "n_fdr_positive": int((significant & (s["mean_z"] > 0)).sum()),
                     "n_fdr_negative": int((significant & (s["mean_z"] < 0)).sum())})
    return pd.DataFrame(rows)


def voxel_networks(mask_img: nib.Nifti1Image, atlas_path: str, centroids_csv: str) -> np.ndarray:
    """Yeo-7 network name of every mask voxel, in mask order (NaN where no parcel)."""
    atlas = resample_to_img(nib.load(atlas_path), mask_img, interpolation="nearest", force_resample=True,
                            copy_header=True).get_fdata().astype(int)[mask_img.get_fdata() > 0]
    parcel_network = pd.read_csv(centroids_csv).set_index("ROI Label")["ROI Name"].str.split("_").str[2]
    return parcel_network.reindex(atlas).to_numpy()


def network_isc(subject_z: dict[str, np.ndarray], voxel_network: np.ndarray, networks: list[str],
                counts: np.ndarray, ci: float) -> pd.DataFrame:
    """Group ISC per network: each subject's mean over the network's voxels, then a bootstrap CI over subjects."""
    rows = []
    for network in networks:
        in_network = voxel_network == network
        for name, z in subject_z.items():
            per_subject = z[:, in_network].mean(axis=1, keepdims=True)  # (subjects, 1)
            boot = counts @ per_subject / len(per_subject)
            low, high = np.percentile(boot, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
            rows.append({"network": network, "map": name, "n_voxels": int(in_network.sum()),
                         "mean_z": float(per_subject.mean()), "ci_low_z": float(low), "ci_high_z": float(high)})
    return pd.DataFrame(rows)


def to_nifti(values: np.ndarray, mask: np.ndarray, affine: np.ndarray) -> nib.Nifti1Image:
    """(voxels,) or (maps, voxels) on mask's True voxels -> 3D or 4D image, NaN elsewhere."""
    values = np.atleast_2d(values)
    volume = np.full(mask.shape + (len(values),), np.nan, dtype=np.float32)
    volume[mask] = values.T
    return nib.Nifti1Image(volume[..., 0] if len(values) == 1 else volume, affine)


def order_dir(cfg: DictConfig, order: str) -> Path:
    """`<qc_root>/<source>/<output_subdir>/order_<order>`."""
    return Path(cfg.qc_root) / cfg.source / cfg.output_subdir / f"order_{order}"


def run_order(cfg: DictConfig, order: str) -> Path:
    """Voxel-wise ISC of one order, raw vs cfg.source, written to `order_dir`; returns that folder."""
    out_dir = order_dir(cfg, order)
    check_output_root(out_dir, cfg.overwrite)
    dirs = {d: out_dir / d for d in ("subject_maps", "group_maps", "tables")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    segments = paired_segments(cfg, order)
    check_headers(segments)
    mask_img = mask_on_grid(cfg.mask, segments["path_raw"].iloc[0])
    mask = mask_img.get_fdata() > 0
    log.info("order %s: %d segments, %d subjects, %d mask voxels", order, len(segments),
             segments["subject"].nunique(), mask.sum())

    subject_z, valid, qc = {}, [], []
    for name in DATASETS:
        scans, dataset_valid = load_dataset(segments, f"path_{name}", mask)
        subjects, subject_z[name], dataset_qc = loso_isc(scans)
        valid.append(dataset_valid)
        qc.append(dataset_qc.assign(dataset=name))

    # compare only voxels with signal in every scan of both datasets
    common = np.logical_and.reduce(valid)
    common_mask = mask.copy()
    common_mask[mask] = common
    z = {name: zs[:, common] for name, zs in subject_z.items()}
    z[DIFF] = z["corrected"] - z["raw"]  # paired: same subjects in the same order
    log.info("order %s: common mask %d of %d voxels", order, common.sum(), mask.sum())

    affine = mask_img.affine
    nib.save(nib.Nifti1Image(common_mask.astype(np.uint8), affine), out_dir / "common_mask.nii.gz")
    pd.Series(subjects, name="subject").to_csv(dirs["subject_maps"] / "subjects.csv", index_label="volume")
    for name in DATASETS:
        nib.save(to_nifti(z[name], common_mask, affine), dirs["subject_maps"] / f"subject_isc_z_{name}.nii.gz")

    counts = bootstrap_counts(len(subjects), cfg.bootstrap.n_boot, cfg.bootstrap.seed)
    stats = {name: group_statistics(zs, counts, cfg.bootstrap.ci, cfg.fdr_alpha) for name, zs in z.items()}
    for name, s in stats.items():
        for key in GROUP_MAPS:
            nib.save(to_nifti(s[key], common_mask, affine), dirs["group_maps"] / f"{name}_{key}.nii.gz")

    segments.drop(columns=["path_raw", "path_corrected"]).to_csv(dirs["tables"] / "segments.csv", index=False)
    pd.concat(qc).to_csv(dirs["tables"] / "scan_qc.csv", index=False)
    n_scans = segments.groupby("subject").size().reindex(subjects).to_numpy()
    pd.DataFrame({"subject": subjects, "n_scans": n_scans,
                  **{f"median_z_{name}": np.median(zs, axis=1) for name, zs in z.items()}}
                 ).to_csv(dirs["tables"] / "subject_summary.csv", index=False)
    summary = group_summary(stats, cfg.fdr_alpha)
    summary.to_csv(dirs["tables"] / "group_summary.csv", index=False)
    voxel_network = voxel_networks(nib.Nifti1Image(common_mask.astype(np.uint8), affine), cfg.atlas,
                                   cfg.centroids_csv)
    network_isc(z, voxel_network, list(cfg.networks), counts, cfg.bootstrap.ci).to_csv(
        dirs["tables"] / "network_isc.csv", index=False)

    manifest = {"source": cfg.source, "order": order, "video": segments["video"].iloc[0],
                "segments_csv": str(cfg.segments_csv),
                "segments_csv_sha256": hashlib.sha256(Path(cfg.segments_csv).read_bytes()).hexdigest(),
                "n_subjects": len(subjects), "n_segments": len(segments), "n_voxels": int(common.sum()),
                "maps": list(z), "bootstrap": OmegaConf.to_container(cfg.bootstrap), "fdr_alpha": cfg.fdr_alpha}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("order %s done -> %s\n%s", order, out_dir, summary.round(4).to_string(index=False))
    return out_dir
