"""Losses on stitched chunk sequences: temporal consistency and FC preservation."""
import logging

import torch
import torch.nn.functional as F
from torch import Tensor

log = logging.getLogger(__name__)

# below this std (PSC units) an ROI is degenerate, e.g. entirely in zeroed background
MIN_ROI_STD = 1e-4


def temporal_consistency_loss(input_seq: Tensor, corrected_seq: Tensor) -> Tensor:
    """Keep the volume-to-volume changes of the corrected sequence close to the input's.

    Args:
        input_seq: (S, T, X, Y, Z) S consecutive chunks of the input.
        corrected_seq: (S, T, X, Y, Z) their corrections.

    Returns:
        L1 between first temporal differences, computed on the stitched S*T volumes so the
        jumps at chunk boundaries are penalised too.
    """
    S, T = input_seq.shape[:2]
    x = input_seq.reshape(S * T, *input_seq.shape[2:]).detach()
    y = corrected_seq.reshape(S * T, *corrected_seq.shape[2:])
    return F.l1_loss(y[1:] - y[:-1], x[1:] - x[:-1])


def pearson_corr_matrix(X: Tensor) -> Tensor:
    """Differentiable Pearson correlation between columns.

    Args:
        X: (T, N) time series of N ROIs.

    Returns:
        (N, N) correlation; near-constant ROIs get 0 off-diagonal and 1 on the diagonal instead of NaN.
    """
    X_centered = X - X.mean(dim=0, keepdim=True)
    std = X_centered.std(dim=0, keepdim=True)
    degenerate = std.squeeze(0) < MIN_ROI_STD
    X_norm = (X_centered / std.clamp(min=MIN_ROI_STD)).masked_fill(degenerate.unsqueeze(0), 0.0)
    corr = (X_norm.T @ X_norm) / max(X.shape[0] - 1, 1)
    # out-of-place so autograd never sees an in-place write
    return corr + torch.diag(degenerate.to(corr.dtype))


def check_fc_matrix(fc: Tensor, name: str = "fc", tol: float = 1e-2) -> None:
    """Assert fc is square, symmetric, with unit diagonal."""
    assert fc.dim() == 2 and fc.shape[0] == fc.shape[1], f"{name}: expected (N, N), got {tuple(fc.shape)}"
    asym = (fc - fc.T).abs().max().item()
    assert asym < tol, f"{name}: not symmetric (max |fc - fc.T| = {asym:.2e})"
    diag_err = (torch.diagonal(fc) - 1.0).abs().max().item()
    assert diag_err < tol, f"{name}: diagonal not ~1 (max error {diag_err:.2e})"


def fc_strength_mask(fc_reference: Tensor, strategy: str, threshold: float = 0.3, top_k: int | None = None,
                     percentile: float | None = None) -> Tensor:
    """Select the strong connections of a reference FC matrix by |r|.

    Args:
        fc_reference: (N, N) correlation matrix.
        strategy: "threshold" (|r| >= threshold), "topk" (k strongest) or "percentile".
        threshold: Cutoff for "threshold".
        top_k: Number of pairs for "topk".
        percentile: 0-100 cutoff for "percentile".

    Returns:
        (N, N) bool mask, upper triangle only so each pair counts once.

    Raises:
        ValueError: If the strategy is unknown or its parameter is missing.
    """
    n = fc_reference.shape[0]
    off_diag = torch.triu(torch.ones(n, n, device=fc_reference.device, dtype=torch.bool), diagonal=1)
    strength = fc_reference.abs()
    if strategy == "threshold":
        cutoff = threshold
    elif strategy == "topk":
        if top_k is None:
            raise ValueError("fc mask strategy 'topk' needs top_k")
        values = strength[off_diag]
        k = min(top_k, values.numel())
        if k == 0:
            return torch.zeros_like(off_diag)
        cutoff = torch.topk(values, k).values.min()
    elif strategy == "percentile":
        if percentile is None:
            raise ValueError("fc mask strategy 'percentile' needs percentile")
        cutoff = torch.quantile(strength[off_diag], percentile / 100.0)
    else:
        raise ValueError(f"unknown fc mask strategy {strategy!r}")
    return (strength >= cutoff) & off_diag


def fc_loss(input_roi_ts: Tensor, corrected_roi_ts: Tensor, mask_strategy: str = "threshold",
            threshold: float = 0.3, top_k: int | None = None, percentile: float | None = None,
            stats: dict | None = None) -> Tensor:
    """L1 between input and corrected FC over the input's strong connections only.

    The mask comes from the corrupted input, the subject's actual (noisy) network; weak connections
    are left free to change.

    Args:
        input_roi_ts: (T_total, n_rois) ROI time series of the input.
        corrected_roi_ts: (T_total, n_rois) ROI time series of the correction.
        mask_strategy, threshold, top_k, percentile: Passed to fc_strength_mask.
        stats: If given, filled with n_retained, n_total_pairs and retained_fraction.

    Returns:
        Scalar loss; 0 if the mask keeps no pairs.
    """
    assert input_roi_ts.shape == corrected_roi_ts.shape
    fc_inp = pearson_corr_matrix(input_roi_ts)
    fc_cor = pearson_corr_matrix(corrected_roi_ts)
    check_fc_matrix(fc_inp.detach(), "fc_input")
    check_fc_matrix(fc_cor.detach(), "fc_corrected")

    mask = fc_strength_mask(fc_inp.detach(), mask_strategy, threshold, top_k, percentile)
    n = fc_inp.shape[0]
    n_pairs = n * (n - 1) // 2
    n_retained = int(mask.sum().item())
    if stats is not None:
        stats.update(n_retained=n_retained, n_total_pairs=n_pairs, retained_fraction=n_retained / max(n_pairs, 1))
    if n_retained == 0:
        log.warning("fc_loss: mask retained 0 ROI pairs, returning zero loss")
        return torch.zeros((), device=fc_inp.device, dtype=fc_inp.dtype)
    return F.l1_loss(fc_cor[mask], fc_inp[mask].detach())
