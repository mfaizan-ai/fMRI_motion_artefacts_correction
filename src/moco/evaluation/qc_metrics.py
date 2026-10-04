"""Whole-run and group-level denoising QC: DVARS, tSNR, QC-FC, its distance dependence, FC significance, modularity."""
import nibabel as nib
import numpy as np
from netneurotools.modularity import consensus_modularity
from nilearn import plotting
from scipy import stats
from statsmodels.stats.multitest import multipletests

FISHER_EPS = 1e-7  # keeps arctanh finite on the unit diagonal / perfectly correlated edges
R_EPS = 1e-12  # keeps the QC-FC t statistic finite at |r| = 1


def dvars_timeseries(func: np.ndarray, mask: np.ndarray, intensity_normalization: float,
                     variance_tol: float) -> np.ndarray:
    """Non-standardised DVARS per frame, nipype's `compute_dvars` formula on in-memory arrays.

    Args:
        func: (X, Y, Z, T) run in BOLD units.
        mask: (X, Y, Z) brain mask.
        intensity_normalization: Brain median is scaled to this value first (nipype default 1000).
        variance_tol: Voxels whose robust SD is at or below this are dropped.

    Returns:
        (T-1,) DVARS.
    """
    voxels = func[mask]
    if intensity_normalization != 0:
        voxels = voxels / np.median(voxels) * intensity_normalization
    # robust SD = IQR / 1.349; "lower" interpolation matches FSL and nipype
    robust_sd = (np.percentile(voxels, 75, axis=1, method="lower")
                 - np.percentile(voxels, 25, axis=1, method="lower")) / 1.349
    voxels = voxels[robust_sd > variance_tol]
    return np.sqrt(np.square(np.diff(voxels, axis=1)).mean(axis=0))


def tsnr(func: np.ndarray, mask: np.ndarray, std_floor: float) -> float:
    """Mean over brain voxels of temporal mean / temporal std, nipype's `TSNR` convention.

    Args:
        func: (X, Y, Z, T) run in BOLD units.
        mask: (X, Y, Z) brain mask.
        std_floor: Voxels with temporal std at or below this get tSNR 0 (nipype uses 1e-3).
    """
    mean_img, std_img = func.mean(axis=-1), func.std(axis=-1)
    tsnr_img = np.zeros_like(mean_img)
    valid = std_img > std_floor
    tsnr_img[valid] = mean_img[valid] / std_img[valid]
    return float(tsnr_img[mask].mean())


def fisher_z(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, -1 + FISHER_EPS, 1 - FISHER_EPS))


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """(R, R) -> (R*(R-1)/2,) edges above the diagonal."""
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def edges_to_matrix(edges: np.ndarray, n_rois: int, fill: float | bool = np.nan) -> np.ndarray:
    """Inverse of `upper_triangle`: symmetric (R, R) matrix with `fill` on the diagonal."""
    matrix = np.full((n_rois, n_rois), fill, dtype=edges.dtype)
    iu = np.triu_indices(n_rois, k=1)
    matrix[iu] = edges
    matrix.T[iu] = edges
    return matrix


def fdr(p: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg over the finite p-values.

    Returns:
        (significant, q): bool mask and adjusted p-values; non-finite p stay not significant / NaN.
    """
    significant = np.zeros(p.shape, dtype=bool)
    q = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    significant[valid], q[valid], _, _ = multipletests(p[valid], alpha=alpha, method="fdr_bh")
    return significant, q


def qc_fc(edges: np.ndarray, mean_fd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Correlation of each edge's FC with subject mean FD (QC-FC).

    Args:
        edges: (S, E) Fisher-z FC per subject.
        mean_fd: (S,) mean framewise displacement per subject.

    Returns:
        (r, p): (E,) Pearson r and its two-sided p-value (t-test, df = S-2).

    Raises:
        ValueError: If there are fewer than 3 subjects, non-finite inputs, or FD has no variance.
    """
    n = len(mean_fd)
    if n < 3:
        raise ValueError("QC-FC needs at least 3 subjects")
    if not (np.all(np.isfinite(edges)) and np.all(np.isfinite(mean_fd))):
        raise ValueError("FC edges or mean FD contain NaN/inf")
    if np.std(mean_fd) == 0:
        raise ValueError("mean FD has zero variance across subjects")
    fd_c = mean_fd - mean_fd.mean()
    edges_c = edges - edges.mean(axis=0, keepdims=True)
    r = (edges_c.T @ fd_c) / np.sqrt((edges_c**2).sum(axis=0) * (fd_c**2).sum())
    r_safe = np.clip(r, -1 + R_EPS, 1 - R_EPS)
    t = r_safe * np.sqrt((n - 2) / (1 - r_safe**2))
    return r, 2 * stats.t.sf(np.abs(t), df=n - 2)


def roi_distance_matrix(atlas_path: str) -> np.ndarray:
    """(R, R) Euclidean distance in mm between parcel centres, ROIs in ascending label order."""
    coords = plotting.find_parcellation_cut_coords(nib.load(atlas_path))
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def distance_dependence(qcfc_r: np.ndarray, distance_edges: np.ndarray) -> tuple[float, float]:
    """QC-FC-DD: Pearson correlation of QC-FC r with inter-ROI distance over finite edges."""
    valid = np.isfinite(qcfc_r)
    r, p = stats.pearsonr(distance_edges[valid], qcfc_r[valid])
    return float(r), float(p)


def fc_significance(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-sample two-sided t-test of each edge's Fisher-z FC against 0 across subjects.

    Args:
        edges: (S, E) Fisher-z FC per subject.

    Returns:
        (mean_z, p): (E,) across-subject mean Fisher-z and p-value (df = S-1).
    """
    _, p = stats.ttest_1samp(edges, popmean=0.0, axis=0)
    return edges.mean(axis=0), p


def signed_asymmetric_q(fc: np.ndarray, communities: np.ndarray, gamma: float) -> float:
    """Signed modularity with asymmetric weighting of negative edges (Rubinov & Sporns 2011).

    Args:
        fc: (R, R) connectivity; symmetrised and its diagonal zeroed here.
        communities: (R,) community label per ROI.
        gamma: Resolution parameter.
    """
    w = (fc + fc.T) / 2
    np.fill_diagonal(w, 0)
    w_pos, w_neg = np.maximum(w, 0), np.maximum(-w, 0)
    s_pos, s_neg = w_pos.sum(), w_neg.sum()
    k_pos = w_pos.sum(axis=1)
    b_pos = w_pos - gamma * np.outer(k_pos, k_pos) / s_pos
    if s_neg > 0:
        k_neg = w_neg.sum(axis=1)
        b_neg = w_neg - gamma * np.outer(k_neg, k_neg) / s_neg
    else:
        b_neg = np.zeros_like(w)
    b_signed = b_pos / s_pos - b_neg / (s_pos + s_neg)
    same_module = communities[:, None] == communities[None, :]
    return float(b_signed[same_module].sum())


def consensus_q(fc: np.ndarray, gamma: float, repeats: int, seed: int) -> tuple[float, int]:
    """Consensus Louvain communities (netneurotools, negative_asym) and their signed modularity.

    Args:
        fc: (R, R) correlation matrix.
        gamma: Resolution parameter.
        repeats: Louvain runs entering the consensus.
        seed: Louvain seed.

    Returns:
        (Q, number of communities).
    """
    fc = (fc + fc.T) / 2
    np.fill_diagonal(fc, 0)
    communities, _, _ = consensus_modularity(adjacency=fc, gamma=gamma, B="negative_asym", repeats=repeats,
                                             seed=seed)
    communities = np.asarray(communities).reshape(-1)
    return signed_asymmetric_q(fc, communities, gamma), int(np.unique(communities).size)
